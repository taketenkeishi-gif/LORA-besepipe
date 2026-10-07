"""モデル非依存の学習基盤 — TrainingBackend / TrainingContext / PreparedRun。

ユーザーはモデル（ModelSpec）を選ぶだけでよく、内部でどのツール
（musubi-tuner / kohya_ss 等）が使われるかは TrainingBackend の実装詳細として
隠蔽される。ルーター層（training.py）は ``training_backend`` 名から
TrainingBackend を解決し、prepare -> cache -> train -> export の順で
呼び出すだけのオーケストレーションに徹する。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingContext:
    """1 回の学習 run を実行するために必要な情報一式。"""

    run_id: int
    project_id: int
    model_family: str
    cfg: dict[str, Any]        # training_runs.config_json をパースした辞書
    settings: dict[str, str]   # app_settings（ツールパス等）


@dataclass
class PreparedRun:
    """prepare() の出力。cache()/train() に引き継がれる実行計画。"""

    run_dir: Path
    config_path: Path
    log_path: Path
    cmd: list[str]
    cwd: str
    python_exe: str
    env: dict[str, str] | None = None
    # バックエンド固有の中間データ（キャッシュ生成に使うモデル解決結果等）を
    # 自由に格納できるようにしておく。他バックエンドはここを読まない。
    extra: dict[str, Any] = field(default_factory=dict)


class TrainingBackend(ABC):
    """モデルファミリーが「どのツールで学習されるか」を隠蔽する抽象境界。

    呼び出し側（training.py の _runner_loop 相当）は以下の順で呼ぶ:

        if not backend.is_ready(settings):
            ...（フォールバック判断はルーター側の責務）
        prepared = backend.prepare(ctx)
        if not backend.cache(ctx, prepared):
            ...（失敗処理）
        backend.train(ctx, prepared)   # ブロッキング。監視・DB更新まで内包
        backend.export(ctx, prepared)  # 学習成果物の後処理（既定は no-op）
    """

    #: UI/API に露出してよい表示名（"Kohya" 等の実装名は使わない）
    display_name: str = ""

    @abstractmethod
    def is_ready(self, settings: dict[str, str]) -> bool:
        """このバックエンドを実行できる環境が揃っているか。"""
        ...

    def supports_resume(self) -> bool:
        """このバックエンドが「既存checkpointから継続学習」をサポートするか。

        既定はFalse（未サポート）。学習エンジンが重み継続用の引数
        (kohyaの--network_weights等)を提供しない場合、Resumeを装って
        epoch 0から再学習するとユーザーが「継続された」と誤認するため、
        サポートしないバックエンドは/resumeで明示的エラーにする
        (app.routers.training.resume()が本メソッドを見て判断する)。
        """
        return False

    @abstractmethod
    def prepare(self, ctx: TrainingContext) -> PreparedRun:
        """train_dir 準備・データセット設定・TOML 生成など学習開始前の準備。

        準備に失敗した場合は例外を送出する（呼び出し側が run を error 化する）。
        """
        ...

    def cache(self, ctx: TrainingContext, prepared: PreparedRun) -> bool:
        """学習前のキャッシュ生成ステップ。不要なバックエンドは既定の no-op(True) でよい。"""
        return True

    @abstractmethod
    def train(self, ctx: TrainingContext, prepared: PreparedRun) -> None:
        """学習サブプロセスを起動し、完了まで監視する（ブロッキング）。

        DB への状態反映（training/completed/error）は本メソッドの責務。
        """
        ...

    def export(self, ctx: TrainingContext, prepared: PreparedRun) -> None:
        """学習成果物（チェックポイント等）の後処理。既定は no-op。"""
        return None
