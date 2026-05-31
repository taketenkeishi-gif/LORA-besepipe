"""Pixiv R18 authentication and session management service."""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from playwright.async_api import async_playwright, BrowserContext, Page

from ..db import get_conn

logger = logging.getLogger(__name__)

# Browser popup session cache: session_id -> PixivBrowserSession
BROWSER_SESSIONS: dict[str, "PixivBrowserSession"] = {}

# Persistent browser profile directory — cookies survive across app restarts
_PIXIV_PROFILE_DIR = Path.home() / ".lora_basepipe" / "pixiv_profile"

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def _get_pixiv_cookie_header(phpsessid: str) -> str:
    return f"PHPSESSID={phpsessid}; R18=1; age_confirmation=1"


async def validate_session(phpsessid: str) -> dict:
    """
    Validate Pixiv session using all cookies from persistent profile.
    Pixiv requires full cookie set (cf_clearance, device_token, etc.) — PHPSESSID alone returns 401.
    """
    if not phpsessid:
        return {"is_valid": False, "user_id": None, "user_name": None, "premium": False, "error": "empty_session"}

    # Load all cookies from persistent profile for full authentication
    all_cookies: list[dict] = []
    if _PIXIV_PROFILE_DIR.exists():
        try:
            pw = await async_playwright().start()
            ctx = await pw.chromium.launch_persistent_context(
                user_data_dir=str(_PIXIV_PROFILE_DIR),
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
                user_agent=_BROWSER_UA,
            )
            all_cookies = await ctx.cookies(["https://www.pixiv.net", "https://pixiv.net"])
            await ctx.close()
            await pw.stop()
        except Exception as e:
            logger.warning(f"validate_session: could not load profile cookies: {e}")

    if all_cookies:
        cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in all_cookies)
    else:
        cookie_str = _get_pixiv_cookie_header(phpsessid)

    headers = {
        "User-Agent": _BROWSER_UA,
        "Cookie": cookie_str,
        "Referer": "https://www.pixiv.net/",
        "Accept": "application/json, text/plain, */*",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://www.pixiv.net/ajax/user/self", headers=headers, timeout=20)

        if resp.status_code == 200:
            data = resp.json()
            # Response uses "userData" key (not "body")
            user_data = data.get("userData") or data.get("body") or {}
            if isinstance(user_data, dict) and user_data:
                return {
                    "is_valid": True,
                    "user_id": str(user_data.get("id") or user_data.get("userId", "")),
                    "user_name": user_data.get("name"),
                    "premium": user_data.get("premium", False),
                    "error": None,
                }
            return {"is_valid": False, "user_id": None, "user_name": None, "premium": False, "error": "empty_body"}
        return {"is_valid": False, "user_id": None, "user_name": None, "premium": False, "error": f"http_{resp.status_code}"}
    except Exception as e:
        logger.error(f"validate_session error: {e}")
        return {"is_valid": False, "user_id": None, "user_name": None, "premium": False, "error": str(e)}


def _extract_phpsessid_from_cookies(cookies: list[dict]) -> Optional[str]:
    for cookie in cookies:
        if cookie.get("name") == "PHPSESSID":
            return cookie.get("value")
    return None


def _save_session_to_db(phpsessid: str, user_id: str, expires_at: str) -> bool:
    try:
        conn = get_conn()
        cur = conn.cursor()
        now = datetime.now().isoformat()
        for key, val in [
            ("pixiv_session", phpsessid),
            ("pixiv_session_expires_at", expires_at),
            ("pixiv_session_user_id", user_id),
            ("pixiv_session_last_validated", now),
        ]:
            cur.execute(
                "INSERT INTO app_settings(key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
                (key, val, now, val, now),
            )
        conn.commit()
        conn.close()
        logger.info(f"Session saved for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save session to DB: {e}")
        return False


def _load_session_from_db() -> dict:
    try:
        conn = get_conn()
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", ("pixiv_session",)).fetchone()
        conn.close()
        if row and row["value"]:
            return {"phpsessid": row["value"], "exists": True}
        return {"phpsessid": None, "exists": False}
    except Exception as e:
        logger.error(f"Failed to load session from DB: {e}")
        return {"phpsessid": None, "exists": False}


async def refresh_session_if_expired() -> dict:
    """Check current session; if invalid, open browser for re-login."""
    session = _load_session_from_db()
    phpsessid = session.get("phpsessid")

    if not phpsessid:
        return {"refreshed": False, "is_valid": False, "phpsessid": None, "user_id": None, "error": "no_session"}

    result = await validate_session(phpsessid)

    if result["is_valid"]:
        try:
            conn = get_conn()
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO app_settings(key, value, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
                ("pixiv_session_last_validated", now, now, now, now),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update last_validated: {e}")
        return {"refreshed": False, "is_valid": True, "phpsessid": phpsessid, "user_id": result["user_id"], "error": None}

    # Session expired — try silent re-login via persistent profile
    logger.warning("Session expired; attempting silent re-login via saved browser profile...")
    browser_session = PixivBrowserSession("auto_refresh")
    login_result = await browser_session.try_silent_login()

    if login_result["success"]:
        _save_session_to_db(login_result["phpsessid"], login_result["user_id"], login_result["expires_at"])
        return {"refreshed": True, "is_valid": True, "phpsessid": login_result["phpsessid"], "user_id": login_result["user_id"], "error": None}

    logger.error(f"Re-login failed: {login_result['error']}")
    return {"refreshed": False, "is_valid": False, "phpsessid": None, "user_id": None, "error": f"relogin_failed: {login_result['error']}"}


class PixivBrowserSession:
    """
    Persistent-profile Playwright session for Pixiv login.

    Uses launch_persistent_context() so cookies survive across browser restarts.
    First login: headful browser shown to user.
    Subsequent logins: browser opens silently, existing cookie validated and returned.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.playwright = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.login_result: Optional[dict] = None
        self.created_at = datetime.now()
        self.timeout_seconds = 600

    async def _launch_context(self) -> None:
        """Launch Chromium with persistent profile."""
        _PIXIV_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        self.playwright = await async_playwright().start()
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=str(_PIXIV_PROFILE_DIR),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            user_agent=_BROWSER_UA,
        )
        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = await self.context.new_page()

    async def try_silent_login(self) -> dict:
        """
        Open persistent profile silently and check if PHPSESSID cookie is still present.
        Returns login result without showing UI if cookie is already valid.
        """
        try:
            await self._launch_context()
            cookies = await self.context.cookies()
            phpsessid = _extract_phpsessid_from_cookies(cookies)

            if phpsessid:
                result = await validate_session(phpsessid)
                if result["is_valid"]:
                    expires_at = (datetime.now() + timedelta(days=30)).isoformat()
                    return {
                        "success": True,
                        "phpsessid": phpsessid,
                        "user_id": result["user_id"],
                        "user_name": result["user_name"],
                        "expires_at": expires_at,
                        "error": None,
                    }

            return {"success": False, "phpsessid": None, "user_id": None, "user_name": None, "expires_at": None, "error": "no_valid_cookie"}
        except Exception as e:
            logger.error(f"try_silent_login error: {e}")
            return {"success": False, "phpsessid": None, "user_id": None, "user_name": None, "expires_at": None, "error": str(e)}
        finally:
            await self.close()

    async def start_browser_for_login(self) -> dict:
        """
        Launch persistent-profile browser.
        If valid PHPSESSID already exists, return immediately without showing UI.
        Otherwise open Pixiv login page for the user.
        """
        try:
            await self._launch_context()

            # Check if already authenticated via saved cookie (URL-scoped for reliability)
            cookies = await self.context.cookies(["https://www.pixiv.net", "https://pixiv.net"])
            phpsessid = _extract_phpsessid_from_cookies(cookies)
            if phpsessid:
                result = await validate_session(phpsessid)
                if result["is_valid"]:
                    expires_at = (datetime.now() + timedelta(days=30)).isoformat()
                    self.login_result = {
                        "success": True,
                        "phpsessid": phpsessid,
                        "user_id": result["user_id"],
                        "user_name": result["user_name"],
                        "expires_at": expires_at,
                        "error": None,
                    }
                    await self.close()
                    logger.info(f"Already authenticated as user {result['user_id']}")
                    return {"success": True, "popup_url": None, "already_logged_in": True, "login_result": self.login_result, "error": None}

            # Need user to log in — show login page
            await self.page.goto("https://accounts.pixiv.net/login", wait_until="networkidle")
            logger.info(f"Browser session {self.session_id} opened for login")
            return {
                "success": True,
                "popup_url": f"http://localhost:8000/settings/pixiv/popup/{self.session_id}",
                "already_logged_in": False,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Failed to start browser session: {e}")
            await self.close()
            return {"success": False, "popup_url": None, "error": str(e)}

    async def wait_for_login(self, timeout_ms: int = 600000) -> dict:
        """
        Poll for PHPSESSID cookie in persistent context.
        - Uses URL-scoped cookie query for reliability
        - Saves to DB as soon as cookie appears (before validation)
        - Handles browser close gracefully
        """
        if not self.page or not self.context:
            return {"success": False, "phpsessid": None, "user_id": None, "user_name": None, "expires_at": None, "error": "browser_not_initialized"}

        _PIXIV_URLS = ["https://www.pixiv.net", "https://pixiv.net", "https://accounts.pixiv.net"]

        try:
            logger.info(f"Polling for PHPSESSID cookie (session {self.session_id})...")
            deadline = asyncio.get_event_loop().time() + timeout_ms / 1000

            while asyncio.get_event_loop().time() < deadline:
                try:
                    cookies = await self.context.cookies(_PIXIV_URLS)
                except Exception:
                    # Browser closed by user — context is gone, nothing more to do here
                    logger.info("Browser closed by user during polling")
                    return {"success": False, "phpsessid": None, "user_id": None, "user_name": None, "expires_at": None, "error": "browser_closed"}

                phpsessid = _extract_phpsessid_from_cookies(cookies)
                if phpsessid:
                    logger.info(f"PHPSESSID found, validating...")
                    result = await validate_session(phpsessid)
                    expires_at = (datetime.now() + timedelta(days=30)).isoformat()
                    # Save to DB regardless — even if validate fails, cookie exists
                    user_id = result.get("user_id") or "unknown"
                    _save_session_to_db(phpsessid, user_id, expires_at)

                    self.login_result = {
                        "success": True,
                        "phpsessid": phpsessid,
                        "user_id": user_id,
                        "user_name": result.get("user_name"),
                        "expires_at": expires_at,
                        "error": None,
                    }
                    logger.info(f"Login complete for session {self.session_id}, user {user_id}")
                    return self.login_result

                await asyncio.sleep(2)

            return {"success": False, "phpsessid": None, "user_id": None, "user_name": None, "expires_at": None, "error": "login_timeout"}

        except Exception as e:
            logger.error(f"wait_for_login error: {e}")
            return {"success": False, "phpsessid": None, "user_id": None, "user_name": None, "expires_at": None, "error": str(e)}

    async def close(self):
        try:
            if self.context:
                await self.context.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info(f"Browser session {self.session_id} closed")
        except Exception as e:
            logger.warning(f"Error closing browser session: {e}")

    def is_expired(self) -> bool:
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.timeout_seconds
