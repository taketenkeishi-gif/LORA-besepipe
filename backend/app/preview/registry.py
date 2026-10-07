"""preview_backend 名 -> PreviewProvider インスタンスの解決。"""
from __future__ import annotations

import logging

from .base import PreviewProvider, PreviewRequest, PreviewResult

logger = logging.getLogger(__name__)

_PROVIDERS: dict[str, PreviewProvider] = {}


def register_preview_provider(name: str, provider: PreviewProvider) -> None:
    _PROVIDERS[name] = provider


def get_preview_provider(name: str) -> PreviewProvider | None:
    return _PROVIDERS.get(name)


def list_preview_providers() -> list[str]:
    return list(_PROVIDERS.keys())


def _failure_results(request: PreviewRequest, reason: str) -> list[PreviewResult]:
    """Provider未登録/未接続/例外時のフェイルレスポンスを構築する。

    以前は空リストを返していたため、呼び出し元(_generate_previews_for_checkpoints)の
    last_error変数が空文字列のままとなり、checkpoints.validation_detailに
    「なぜPreviewが失敗したか」が一切記録されない実バグがあった
    (ユーザーはPreview失敗の理由を一切知る術がなかった)。
    prompt_itemsの数だけ、reasonを詰めたPreviewResultを返すことで、
    呼び出し元の集計ロジックを変更せずにerror detailを伝播させる。
    """
    n = max(1, len(request.prompt_items))
    return [PreviewResult(prompt_index=i, seed=0, error=reason) for i in range(n)]


def generate_preview_safe(name: str, request: PreviewRequest) -> list[PreviewResult]:
    """PreviewProvider を解決して呼び出す。

    プレビュー失敗は学習失敗ではない、という方針をこの関数で一元的に保証する:
    Provider 未登録・未接続・生成時の例外はすべてここで捕捉し、呼び出し元
    （学習ループ）には一切例外を伝播させない。ただし失敗理由(error detail)は
    PreviewResult.errorへ必ず格納する(以前は空リストを返し理由が失われていた)。
    """
    provider = get_preview_provider(name)
    if provider is None:
        reason = f"preview provider '{name}' is not registered"
        logger.info("preview: %s のためスキップ", reason)
        return _failure_results(request, reason)
    try:
        if not provider.is_ready(request.settings):
            reason = f"preview provider '{name}' is not ready (未接続または設定不足)"
            logger.info("preview: %s のためスキップ", reason)
            return _failure_results(request, reason)
        return provider.generate_preview(request)
    except Exception as exc:  # noqa: BLE001 — プレビュー失敗は学習失敗にしない
        reason = f"preview provider '{name}' でエラー: {exc}"
        logger.warning("preview: run %s で生成失敗: %s", request.run_id, reason)
        return _failure_results(request, reason)
