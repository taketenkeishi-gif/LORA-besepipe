from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ..db import get_conn
from ..schemas import PreviewPromptsIn, PreviewPromptsOut, ToolPathsIn, ToolPathsOut
from playwright.async_api import async_playwright

from ..services.pixiv_auth import (
    PixivBrowserSession,
    BROWSER_SESSIONS,
    validate_session,
    _save_session_to_db,
    _extract_phpsessid_from_cookies,
    _PIXIV_PROFILE_DIR,
    _BROWSER_UA,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

SETTINGS_KEYS = ("python_exe", "kohya_root", "comfyui_root", "wd14_script", "temp_dir", "dataset_base_dir", "pixiv_session")
PROMPT_KEYS = ("positive_prompt", "negative_prompt")


def _read_paths() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    conn.close()
    found = {str(r["key"]): str(r["value"]) for r in rows}
    return {k: found.get(k, "") for k in SETTINGS_KEYS}


def _read_prompts() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM app_settings WHERE key IN (?, ?)", PROMPT_KEYS).fetchall()
    conn.close()
    found = {str(r["key"]): str(r["value"]) for r in rows}
    return {
        "positive_prompt": found.get("positive_prompt", "masterpiece, best quality, 1girl, portrait"),
        "negative_prompt": found.get("negative_prompt", "low quality, blurry, bad anatomy"),
    }


def _write_paths(payload: dict[str, str]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    for key in SETTINGS_KEYS:
        val = str(payload.get(key, "")).strip()
        cur.execute(
            """
            INSERT INTO app_settings(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, val),
        )
    conn.commit()
    conn.close()


def _write_prompts(payload: dict[str, str]) -> None:
    conn = get_conn()
    cur = conn.cursor()
    for key in PROMPT_KEYS:
        val = str(payload.get(key, "")).strip()
        cur.execute(
            """
            INSERT INTO app_settings(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, val),
        )
    conn.commit()
    conn.close()


def _autodetect() -> dict[str, str]:
    result = {k: "" for k in SETTINGS_KEYS}
    project_root = Path(__file__).resolve().parents[3]
    default_dataset_base = Path(r"C:\ポートフォリオ\SDXL\LoRA_Traning\dataset")

    python_path = shutil.which("python") or ""
    if python_path:
        result["python_exe"] = str(Path(python_path))

    candidates = [
        project_root / "external_tools" / "kohya_ss",
        Path.home() / "kohya_ss",
        Path("C:/kohya_ss"),
        Path("C:/tools/kohya_ss"),
        Path("D:/tools/kohya_ss"),
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            result["kohya_root"] = str(c.resolve())
            break

    comfy_candidates = [
        project_root / "external_tools" / "ComfyUI",
        Path.home() / "ComfyUI",
        Path("C:/ComfyUI"),
        Path("D:/ComfyUI"),
    ]
    for c in comfy_candidates:
        if c.exists() and c.is_dir():
            result["comfyui_root"] = str(c.resolve())
            break

    if result["kohya_root"]:
        wd14 = Path(result["kohya_root"]) / "sd-scripts" / "finetune" / "tag_images_by_wd14_tagger.py"
        if wd14.exists():
            result["wd14_script"] = str(wd14.resolve())
    wd14_local = project_root / "external_tools" / "WD14py" / "tag_images_by_wd14_tagger.py"
    if wd14_local.exists():
        result["wd14_script"] = str(wd14_local.resolve())

    result["temp_dir"] = str((project_root / ".runtime" / "tmp").resolve())
    if default_dataset_base.exists():
        result["dataset_base_dir"] = str(default_dataset_base.resolve())
    else:
        result["dataset_base_dir"] = str((project_root / "external_dataset").resolve())

    return result


def _check_exe(path: str) -> dict:
    if not path:
        return {"ok": False, "reason": "not configured"}
    p = Path(path)
    if not p.exists():
        return {"ok": False, "reason": "path not found"}
    try:
        proc = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=4)
        out = (proc.stdout or proc.stderr or "").strip().splitlines()
        return {"ok": proc.returncode == 0, "reason": out[0] if out else "version check done"}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _check_dir(path: str) -> dict:
    if not path:
        return {"ok": False, "reason": "not configured"}
    p = Path(path)
    if not p.exists():
        return {"ok": False, "reason": "path not found"}
    if not p.is_dir():
        return {"ok": False, "reason": "not a directory"}
    return {"ok": True, "reason": "directory found"}


def _check_file(path: str) -> dict:
    if not path:
        return {"ok": False, "reason": "not configured"}
    p = Path(path)
    if not p.exists():
        return {"ok": False, "reason": "path not found"}
    if not p.is_file():
        return {"ok": False, "reason": "not a file"}
    return {"ok": True, "reason": "file found"}


@router.get("/tool-paths", response_model=ToolPathsOut)
def get_tool_paths() -> ToolPathsOut:
    current = _read_paths()
    detected = _autodetect()
    # 初回は自動検出結果をそのまま初期値として保存
    if all(not v for v in current.values()):
        _write_paths(detected)
        return ToolPathsOut(**_read_paths())
    # 一部だけ空欄の場合も、検出できる値で自動補完する
    merged = current.copy()
    changed = False
    for key in SETTINGS_KEYS:
        if (not merged.get(key)) and detected.get(key):
            merged[key] = detected[key]
            changed = True
    if changed:
        _write_paths(merged)
        return ToolPathsOut(**_read_paths())
    return ToolPathsOut(**current)


@router.put("/tool-paths", response_model=ToolPathsOut)
def update_tool_paths(payload: ToolPathsIn) -> ToolPathsOut:
    _write_paths(payload.model_dump())
    paths = _read_paths()
    _ensure_runtime_dirs(paths)
    return ToolPathsOut(**paths)


def _ensure_runtime_dirs(paths: dict[str, str]) -> None:
    for key in ("temp_dir", "dataset_base_dir"):
        p = paths.get(key, "").strip()
        if not p:
            continue
        Path(p).mkdir(parents=True, exist_ok=True)
    # dataset_base_dir直下に型別フォルダを用意
    base = paths.get("dataset_base_dir", "").strip()
    if base:
        Path(base, "character").mkdir(parents=True, exist_ok=True)
        Path(base, "style").mkdir(parents=True, exist_ok=True)


@router.post("/tool-paths/autodetect", response_model=ToolPathsOut)
def autodetect_tool_paths() -> ToolPathsOut:
    detected = _autodetect()
    _write_paths(detected)
    paths = _read_paths()
    _ensure_runtime_dirs(paths)
    return ToolPathsOut(**paths)


@router.get("/integrations/status")
def integrations_status() -> dict:
    paths = _read_paths()
    checks = {
        "python_exe": _check_exe(paths["python_exe"]),
        "kohya_root": _check_dir(paths["kohya_root"]),
        "comfyui_root": _check_dir(paths["comfyui_root"]),
        "wd14_script": _check_file(paths["wd14_script"]),
        "temp_dir": _check_dir(paths["temp_dir"]),
        "dataset_base_dir": _check_dir(paths["dataset_base_dir"]),
    }
    return {"paths": paths, "checks": checks}


@router.get("/preview-prompts", response_model=PreviewPromptsOut)
def get_preview_prompts() -> PreviewPromptsOut:
    prompts = _read_prompts()
    return PreviewPromptsOut(**prompts)


@router.put("/preview-prompts", response_model=PreviewPromptsOut)
def update_preview_prompts(payload: PreviewPromptsIn) -> PreviewPromptsOut:
    _write_prompts(payload.model_dump())
    return PreviewPromptsOut(**_read_prompts())


# Pixiv Authentication Endpoints


async def _check_profile_cookie() -> dict | None:
    """
    Silently re-launch persistent profile to check for saved PHPSESSID.
    Called after user manually closes the Playwright browser.
    """
    pw = None
    ctx = None
    try:
        pw = await async_playwright().start()
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(_PIXIV_PROFILE_DIR),
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
            user_agent=_BROWSER_UA,
        )
        cookies = await ctx.cookies(["https://www.pixiv.net", "https://pixiv.net"])
        phpsessid = _extract_phpsessid_from_cookies(cookies)
        if phpsessid:
            result = await validate_session(phpsessid)
            user_id = result.get("user_id") or "unknown"
            expires_at = (datetime.now() + timedelta(days=30)).isoformat()
            _save_session_to_db(phpsessid, user_id, expires_at)
            return {"phpsessid": phpsessid, "user_id": user_id}
        return None
    except Exception as e:
        logger.warning(f"_check_profile_cookie error: {e}")
        return None
    finally:
        if ctx:
            try:
                await ctx.close()
            except Exception:
                pass
        if pw:
            try:
                await pw.stop()
            except Exception:
                pass


def _load_pixiv_session_from_db() -> dict:
    """Load Pixiv session data from database."""
    conn = get_conn()
    try:
        phpsessid = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            ("pixiv_session",)
        ).fetchone()
        expires_at = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            ("pixiv_session_expires_at",)
        ).fetchone()
        user_id = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            ("pixiv_session_user_id",)
        ).fetchone()
        last_validated = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            ("pixiv_session_last_validated",)
        ).fetchone()
    finally:
        conn.close()

    return {
        "phpsessid": (phpsessid["value"] if phpsessid and phpsessid["value"] else None),
        "expires_at": (expires_at["value"] if expires_at and expires_at["value"] else None),
        "user_id": (user_id["value"] if user_id and user_id["value"] else None),
        "last_validated": (last_validated["value"] if last_validated and last_validated["value"] else None),
    }


def _save_pixiv_session_to_db(phpsessid: str, user_id: str, expires_at: str) -> bool:
    """Save Pixiv session to database."""
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    try:
        for key, value in [
            ("pixiv_session", phpsessid),
            ("pixiv_session_user_id", user_id),
            ("pixiv_session_expires_at", expires_at),
            ("pixiv_session_last_validated", now),
        ]:
            cur.execute(
                """
                INSERT INTO app_settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
                """,
                (key, value, now, value, now)
            )
        conn.commit()
        logger.info(f"Pixiv session saved for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save Pixiv session: {e}")
        return False
    finally:
        conn.close()


@router.post("/pixiv/login-start")
async def pixiv_login_start() -> dict:
    """
    Start Pixiv login flow as a background task.
    Opens Playwright browser; frontend polls /status every 30s to detect completion.
    """
    import asyncio

    session_id = str(uuid.uuid4())
    browser_session = PixivBrowserSession(session_id)
    BROWSER_SESSIONS[session_id] = browser_session

    async def _run_login():
        result = await browser_session.start_browser_for_login()
        if not result["success"]:
            logger.error(f"Browser start failed: {result['error']}")
            BROWSER_SESSIONS.pop(session_id, None)
            return

        # Already have a valid saved cookie — save and done
        if result.get("already_logged_in") and result.get("login_result"):
            lr = result["login_result"]
            _save_pixiv_session_to_db(lr["phpsessid"], lr["user_id"], lr["expires_at"])
            logger.info(f"Auto-authenticated user {lr['user_id']} from saved profile")
            BROWSER_SESSIONS.pop(session_id, None)
            return

        # Wait for user to log in (cookie is saved to DB inside wait_for_login)
        login_result = await browser_session.wait_for_login()
        await browser_session.close()
        BROWSER_SESSIONS.pop(session_id, None)

        if login_result["success"]:
            logger.info(f"Login complete, user {login_result['user_id']} saved to DB")
        elif login_result.get("error") == "browser_closed":
            # User closed browser manually — silently re-check profile for saved cookie
            logger.info("Browser closed; checking saved profile for cookie...")
            saved = await _check_profile_cookie()
            if saved:
                logger.info(f"Cookie recovered from saved profile: user {saved['user_id']}")
            else:
                logger.warning("No valid cookie found after browser close")
        else:
            logger.warning(f"Login failed: {login_result['error']}")

    asyncio.create_task(_run_login())

    return {
        "success": True,
        "session_id": session_id,
        "popup_url": None,
        "already_logged_in": False,
        "message": "ブラウザが開きます。Pixiv にログインしてください。ログイン後に自動的に反映されます。"
    }


@router.get("/pixiv/popup/{session_id}", response_class=HTMLResponse)
async def pixiv_popup(session_id: str) -> str:
    """Render popup page for Pixiv login (displayed in browser)."""
    browser_session = BROWSER_SESSIONS.get(session_id)

    if not browser_session:
        return """
        <html>
            <head><title>Pixiv Login</title></head>
            <body>
                <h1>Error</h1>
                <p>Session not found. Please start login again.</p>
            </body>
        </html>
        """

    if browser_session.is_expired():
        del BROWSER_SESSIONS[session_id]
        await browser_session.close()
        return """
        <html>
            <head><title>Pixiv Login</title></head>
            <body>
                <h1>Session Expired</h1>
                <p>Login session timed out. Please try again.</p>
            </body>
        </html>
        """

    # Wait for login completion, then close the Playwright browser
    login_result = await browser_session.wait_for_login()
    await browser_session.close()
    BROWSER_SESSIONS.pop(session_id, None)

    if login_result["success"]:
        # Save to DB
        _save_pixiv_session_to_db(
            login_result["phpsessid"],
            login_result["user_id"],
            login_result["expires_at"]
        )

        return f"""
        <html>
            <head>
                <title>Pixiv Login - Success</title>
                <script>
                    window.close();
                </script>
            </head>
            <body>
                <h1>✅ ログイン成功</h1>
                <p>このウィンドウは自動的に閉じます...</p>
            </body>
        </html>
        """
    else:
        return f"""
        <html>
            <head>
                <title>Pixiv Login - Failed</title>
            </head>
            <body>
                <h1>❌ ログイン失敗</h1>
                <p>エラー: {login_result['error']}</p>
                <button onclick="window.close()">ウィンドウを閉じる</button>
            </body>
        </html>
        """


@router.get("/pixiv/status")
async def pixiv_status() -> dict:
    """Get current Pixiv session status."""
    session = _load_pixiv_session_from_db()
    phpsessid = session.get("phpsessid")

    if not phpsessid:
        return {
            "is_logged_in": False,
            "is_valid": False,
            "user_id": None,
            "user_name": None,
            "expires_at": None,
            "last_validated": None,
            "auto_refresh_interval": "1h",
            "message": "ログインしていません"
        }

    # Validate current session
    result = await validate_session(phpsessid)

    return {
        "is_logged_in": True,
        "is_valid": result["is_valid"],
        "user_id": session.get("user_id") or result.get("user_id"),
        "user_name": result.get("user_name"),
        "expires_at": session.get("expires_at"),
        "last_validated": session.get("last_validated"),
        "auto_refresh_interval": "1h",
        "message": "ログイン済み" if result["is_valid"] else "セッション期限切れ"
    }


@router.post("/pixiv/refresh")
async def pixiv_refresh() -> dict:
    """Manually refresh Pixiv session."""
    session = _load_pixiv_session_from_db()
    phpsessid = session.get("phpsessid")

    if not phpsessid:
        logger.warning("No Pixiv session found for refresh")
        return {
            "success": False,
            "refreshed": False,
            "message": "ログインセッションが見つかりません"
        }

    # Validate current session
    result = await validate_session(phpsessid)

    if result["is_valid"]:
        # Update last_validated timestamp
        conn = get_conn()
        now = datetime.now().isoformat()
        try:
            conn.execute(
                """
                INSERT INTO app_settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
                """,
                ("pixiv_session_last_validated", now, now, now, now)
            )
            conn.commit()
        finally:
            conn.close()

        logger.info("Pixiv session is still valid")
        return {
            "success": True,
            "refreshed": False,
            "message": "セッションはまだ有効です"
        }

    # Session expired; launch browser for re-login in background
    logger.warning("Pixiv session expired; launching re-login browser...")
    import asyncio

    async def _relogin():
        session_id = str(uuid.uuid4())
        browser_session = PixivBrowserSession(session_id)
        result = await browser_session.start_browser_for_login()
        if not result["success"]:
            return
        if result.get("already_logged_in") and result.get("login_result"):
            lr = result["login_result"]
            _save_pixiv_session_to_db(lr["phpsessid"], lr["user_id"], lr["expires_at"])
            return
        login_result = await browser_session.wait_for_login()
        await browser_session.close()
        if login_result["success"]:
            _save_pixiv_session_to_db(login_result["phpsessid"], login_result["user_id"], login_result["expires_at"])

    asyncio.create_task(_relogin())
    return {
        "success": True,
        "refreshed": False,
        "message": "ブラウザを起動しました。Pixiv にログインしてください。"
    }


@router.post("/pixiv/logout")
def pixiv_logout() -> dict:
    """Logout and clear Pixiv session."""
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    try:
        for key in ["pixiv_session", "pixiv_session_expires_at", "pixiv_session_user_id", "pixiv_session_last_validated"]:
            cur.execute(
                """
                INSERT INTO app_settings(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
                """,
                (key, "", now, "", now)
            )
        conn.commit()
        logger.info("Pixiv session cleared")
        return {
            "success": True,
            "message": "ログアウトしました"
        }
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        return {
            "success": False,
            "message": f"ログアウト失敗: {e}"
        }
    finally:
        conn.close()
