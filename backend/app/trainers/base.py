from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RequiredModel:
    """モデルが必要とする外部ファイル（チェックポイント、VAE、TE など）の定義。"""

    key: str
    display_name: str
    auto_search_paths: list[str] = field(default_factory=list)
    required: bool = True
    cfg_key: str = ""       # cfg から読む際のキー名（省略時は "{key}_path"）
    toml_key: str = ""      # セット時 → TOML に { toml_key: resolved_path } を挿入
    cache_arg: str = ""     # キャッシュスクリプトへの CLI 引数 "--vae" など
    download_url: str = ""
    hf_repo: str = ""
    filename: str = ""
    sha256: str = ""


@dataclass
class CacheStep:
    """学習前に実行するキャッシュ生成ステップの定義。"""

    script: str             # "krea2_cache_latents.py"
    model_key: str          # → RequiredModel.key（パス解決に使う）
    extra_args: list[str] = field(default_factory=list)
    label: str = ""


@dataclass
class ModelSpec:
    """モデルファミリー固有のすべての設定を保持するデータクラス。

    BaseTrainer は ModelSpec からモデル固有情報を読み取り、
    実行ロジックは一切 if model_family == ... を書かない。

    training_backend / preview_backend は「モデル非依存の学習・プレビュー基盤」
    (app.training / app.preview) が使うルーティングキーであり、UI・API には
    このキーの実装詳細（musubi-tuner / kohya_ss 等の具体名）を露出しない。
    """

    model_family: str
    display_name: str
    backend: str             # "musubi" | "kohya"（後方互換のため維持。内部識別のみに使用）
    train_script: str
    network_module: str
    required_models: list[RequiredModel] = field(default_factory=list)
    cache_steps: list[CacheStep] = field(default_factory=list)
    extra_toml: dict[str, Any] = field(default_factory=dict)
    supports: set[str] = field(default_factory=set)
    default_resolution: int = 512
    default_preset: dict[str, Any] = field(default_factory=dict)
    checkpoint_patterns: list[str] = field(default_factory=list)
    # Preview生成時に検索するcheckpointファイル名パターン。空ならcheckpoint_patterns
    # (Training用)を引き継ぐ(後方互換)。TrainingとPreviewで異なるbase modelを
    # 使い分けたい場合(例: 学習はKrea2 RAW、Previewは高速なKrea2 Turbo)に、
    # if model_family==...の分岐を増やさずこのフィールドだけで表現する。
    # 実Runtime検証(2026-07-05): Krea2 RAWで学習したLoRAはKrea2 Turboへロード・
    # サンプリング・実画像生成まで成功することを確認済み(23秒 vs RAW単体444秒、
    # 同一GPU上でTrainingと同時実行してもRAW同士より競合が大幅に軽減される)。
    preview_checkpoint_patterns: list[str] = field(default_factory=list)
    # プレビュー（サンプル画像）生成の既定パラメータ。モデル世代ごとに適切な値が異なる
    # （例: SD/SDXL は cfg≈7 / euler_a、Flux・Qwen 系は cfg≈1 / 専用サンプラー）。
    preview_sampler: str = "euler_a"
    preview_cfg: float = 7.0
    preview_steps: int = 20
    # Preview生成時のUNET重みdtype（ComfyUI UNETLoaderのweight_dtype）。
    # "default"（既定・無変換）以外を指定するのは、base checkpointの実ファイルサイズが
    # 単体でVRAM容量を超える大型DiT系モデル（例: RAW MMDiT）のみを想定した
    # Architecture固有の設定値であり、if model_family==... の分岐ではなくこのフィールドで表現する。
    preview_unet_weight_dtype: str = "default"
    # Preview生成の上限解像度（None=上限なし、グローバル既定値をそのまま使う）。
    # 学習解像度(default_resolution)とは独立して、大型モデルのPreview生成専用に
    # 現実的な時間で完了する解像度へ制限するためのフィールド。
    preview_max_resolution: int | None = None
    # ── モデル非依存基盤（app.training / app.preview）向けルーティングキー ──
    # 空文字なら後方互換のため backend の値を引き継ぐ（sd/sdxl のみ "sdxl" に読み替え）。
    training_backend: str = ""
    preview_backend: str = "comfyui"

    def __post_init__(self) -> None:
        if not self.training_backend:
            # kohya バックエンドの2モデル(sd/sdxl)は SDXLBackend という
            # モデル非依存の名前の下に統合される（Kohya という実装名は露出させない）。
            self.training_backend = "sdxl" if self.backend == "kohya" else self.backend


class BaseTrainer(ABC):
    """モデル固有知識を持たない実行ロジック基底クラス。

    サブクラス（MusubiRunner / KohyaRunner）が build_toml の
    バックエンド固有の共通キーを提供する。
    """

    # ── モデルパス解決 ───────────────────────────────────────────────────────

    def resolve_models(
        self, spec: ModelSpec, settings: dict, cfg: dict
    ) -> dict[str, str]:
        """RequiredModel リストを元に実際のファイルパスを解決する。

        優先順:
          1. cfg 内の明示パス（cfg_key が設定されていればそのキー、省略時は "{key}_path"）
          2. comfyui_root 配下の auto_search_paths を順番に探索
          3. required=False なら空文字（必須でなければスキップ）
        """
        resolved: dict[str, str] = {}
        comfyui_root = (settings.get("comfyui_root") or "").strip()

        for rm in spec.required_models:
            cfg_key = rm.cfg_key if rm.cfg_key else f"{rm.key}_path"
            explicit = (cfg.get(cfg_key) or "").strip()
            if explicit and Path(explicit).exists():
                resolved[rm.key] = explicit
                continue

            found = ""
            if comfyui_root:
                for rel in rm.auto_search_paths:
                    candidate = Path(comfyui_root) / rel
                    if candidate.exists():
                        found = str(candidate)
                        break
            if found:
                resolved[rm.key] = found
            elif not rm.required:
                resolved[rm.key] = ""
            # required=True で見つからない場合はキーを入れない（呼び出し側でチェック可）

        return resolved

    # ── TOML 生成 ────────────────────────────────────────────────────────────

    def build_toml(
        self,
        spec: ModelSpec,
        common: dict[str, Any],
        resolved: dict[str, str],
    ) -> dict[str, Any]:
        """共通キー + spec.extra_toml + resolved モデルパス注入 を合成した辞書を返す。"""
        d: dict[str, Any] = {}
        d.update(common)
        d.update(spec.extra_toml)
        for rm in spec.required_models:
            if rm.toml_key and resolved.get(rm.key):
                d[rm.toml_key] = resolved[rm.key]
        return d

    @staticmethod
    def toml_serialize(d: dict[str, Any]) -> str:
        """dict → TOML 形式の文字列（型に応じてクォートや bool 変換を統一処理）。"""

        def q(v: str) -> str:
            return v.replace("\\", "\\\\").replace('"', '\\"')

        lines: list[str] = []
        for k, v in d.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, str):
                lines.append(f'{k} = "{q(v)}"')
            elif isinstance(v, int):
                lines.append(f"{k} = {v}")
            elif isinstance(v, float):
                lines.append(f"{k} = {v}")
        return "\n".join(lines) + "\n"

    # ── キャッシュステップ実行 ────────────────────────────────────────────────

    def run_cache_steps(
        self,
        spec: ModelSpec,
        resolved: dict[str, str],
        *,
        dataset_json_path: str,
        python_exe: str,
        tool_root: str,
        log_path: Path,
        env: dict[str, str] | None = None,
    ) -> bool:
        """spec.cache_steps を順番に実行する。失敗したら False を返す。

        CacheStep.model_key に対応する RequiredModel が解決されていない場合は
        警告ログを出してスキップする（required=False の任意モデルを想定）。
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)

        for step in spec.cache_steps:
            label = step.label or step.script

            # スクリプトパスを探す
            script_path = self._find_helper_script(tool_root, step.script)
            if script_path is None:
                self._log(log_path, f"[CACHE] {label}: スクリプトが見つかりません ({step.script}) — スキップ\n")
                continue

            # モデルパスを解決
            model_path = resolved.get(step.model_key, "")
            if not model_path:
                self._log(log_path, f"[CACHE] {label}: モデルパス未解決 ({step.model_key}) — スキップ\n")
                continue

            # cache_arg を取得
            cache_arg = ""
            for rm in spec.required_models:
                if rm.key == step.model_key:
                    cache_arg = rm.cache_arg
                    break

            cmd: list[str] = [python_exe, script_path,
                               "--dataset_config", dataset_json_path]
            if cache_arg:
                cmd += [cache_arg, model_path]
            cmd += list(step.extra_args)

            ok = self._run_subprocess(label, cmd, log_path=log_path, cwd=tool_root, env=env)
            if not ok:
                return False

        return True

    # ── 内部ユーティリティ ────────────────────────────────────────────────────

    @staticmethod
    def _find_helper_script(tool_root: str, script_name: str) -> str | None:
        p = Path(tool_root)
        for cand in [p / script_name,
                     p / "src" / script_name,
                     p / "src" / "musubi_tuner" / script_name]:
            if cand.exists():
                return str(cand)
        return None

    @staticmethod
    def _log(log_path: Path, msg: str) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg)
        except OSError:
            pass

    def _run_subprocess(
        self,
        label: str,
        cmd: list[str],
        *,
        log_path: Path,
        cwd: str,
        env: dict | None = None,
    ) -> bool:
        try:
            self._log(log_path, f"\n[CACHE] {label} 開始\n[CACHE] cmd: {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
            )
            with open(log_path, "ab") as lf:
                for line in proc.stdout:  # type: ignore[union-attr]
                    lf.write(line)
            proc.wait()
            if proc.returncode != 0:
                self._log(log_path, f"[CACHE] {label} 失敗 (returncode={proc.returncode})\n")
                return False
            self._log(log_path, f"[CACHE] {label} 完了\n")
            return True
        except Exception as exc:
            self._log(log_path, f"[CACHE] {label} エラー: {exc}\n")
            return False

    @abstractmethod
    def common_toml(self, spec: ModelSpec, cfg: dict[str, Any]) -> dict[str, Any]:
        """バックエンド固有の共通 TOML キーを返す。サブクラスで実装。"""
        ...
