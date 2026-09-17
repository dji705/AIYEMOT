"""Headless bridge to chatgpt.com's anonymous chat (Playwright).

Adapted from the CHATGPT project's verified flow (mobile-web composer,
native-setter fill, "Stop generating" completion signal, h4/DIV reply
markup). Constraints:
  - ONLY Playwright's bundled Chromium, headless (never a system browser).
  - The internal chatgpt.com API is NOT used; the UI-level flow is the
    stable contract.
  - Reuses one page across submissions; relaunches if the page dies.

Environment:
  YHM_API_BASE    -- (unused here) kept for symmetry with the app module
  CHAT_TARGET_URL -- default https://chatgpt.com/
  BROWSER_HEADLESS -- "0" for a visible browser (debugging), default headless
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger("aiyemot.chat_bridge")


class ChatError(Exception):
    """Recoverable failure (timeout / network / extraction)."""


TARGET_URL = os.environ.get("CHAT_TARGET_URL", "https://chatgpt.com/").strip() or "https://chatgpt.com/"

# --- JS run inside the page (verified against the anonymous mweb UI) --------

STATE_JS = """
() => {
  const ta = document.querySelector('textarea#mobile-composer-prompt');
  const stopBtn = document.querySelector('button[aria-label="Stop generating"]');
  const sendBtn = document.querySelector('button[aria-label="Send message"]');
  return {
    composerReady: !!(ta && !ta.disabled),
    stopVisible: !!stopBtn,
    sendReady: !!(sendBtn && !sendBtn.disabled),
  };
}
"""

FILL_JS = """
(text) => {
  const ta = document.querySelector('textarea#mobile-composer-prompt');
  if (!ta) return false;
  const setter = Object.getOwnPropertyDescriptor(
    HTMLTextAreaElement.prototype, 'value'
  ).set;
  setter.call(ta, text);
  ta.dispatchEvent(new Event('input', { bubbles: true }));
  return true;
}
"""

LAST_REPLY_JS = """
() => {
  // Hidden <h4> labels ("You said:" / "ChatGPT said:") precede sibling DIVs
  // with the message text; the LAST "ChatGPT said:" block is the newest reply.
  const h4s = Array.from(document.querySelectorAll('h4'));
  for (let i = h4s.length - 1; i >= 0; i--) {
    if ((h4s[i].textContent || '').includes('ChatGPT said:')) {
      const target = h4s[i].nextElementSibling;
      return target ? (target.innerText || target.textContent || '') : '';
    }
  }
  return '';
}
"""


class PlaywrightChatBridge:
    """One bundled-Chromium page reused across messages. Thread-safe."""

    def __init__(
        self,
        request_timeout_s: int = 30,
        response_timeout_s: int = 180,
        target_url: str = TARGET_URL,
    ) -> None:
        self.request_timeout_s = request_timeout_s
        self.response_timeout_s = response_timeout_s
        self.target_url = target_url
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- lifecycle
    def _ensure_page(self):
        if self._page is not None:
            try:
                self._page.evaluate("1")  # liveness probe
                return self._page
            except Exception:
                logger.info("page died; relaunching bundled Chromium")
                self._teardown()
        from playwright.sync_api import sync_playwright

        if self._pw is None:
            self._pw = sync_playwright().start()
        if self._browser is None:
            headless = os.environ.get("BROWSER_HEADLESS", "1") not in ("0", "false", "no")
            self._browser = self._pw.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ],
            )
        self._context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"
            ),
            viewport={"width": 412, "height": 915},
            locale="he-IL",
        )
        self._context.add_cookies(
            [
                {
                    "name": "oai-mweb-route-desktop",
                    "value": "1",
                    "domain": ".chatgpt.com",
                    "path": "/",
                }
            ]
        )
        self._page = self._context.new_page()
        self.new_conversation()
        return self._page

    def new_conversation(self) -> None:
        """Fresh anonymous conversation (base URL reload)."""
        timeout = self.request_timeout_s * 1000
        try:
            self._page.goto(
                self.target_url, wait_until="domcontentloaded", timeout=timeout
            )
            self._page.wait_for_selector(
                "textarea#mobile-composer-prompt", timeout=timeout
            )
        except ChatError:
            raise
        except Exception as exc:
            raise ChatError(f"could not load {self.target_url}: {exc}") from exc

    def reset_conversation(self) -> None:
        """Start a fresh anonymous conversation (LOGOUT / new session).

        Non-blocking for callers: safe to invoke before the next submission;
        the next send_and_wait continues from a brand-new conversation.
        """
        with self._lock:
            if self._page is None:
                return  # nothing open yet; the next send starts fresh anyway
            try:
                self.new_conversation()
                logger.info("conversation reset (new anonymous session)")
            except Exception as exc:
                logger.warning("reset_conversation failed (%s); page will be relaunched", exc)
                self._teardown()

    def _teardown(self) -> None:
        for closer in (
            lambda: self._context and self._context.close(),
            lambda: self._browser and self._browser.close(),
            lambda: self._pw and self._pw.stop(),
        ):
            try:
                closer()
            except Exception:
                pass
        self._page = None
        self._context = None
        self._browser = None
        self._pw = None

    def close(self) -> None:
        with self._lock:
            self._teardown()

    # ---------------------------------------------------------- conversation
    def send_and_wait(self, text: str) -> str:
        """Submit `text` and return ONLY the finished assistant reply."""
        if not text.strip():
            raise ChatError("empty message")
        with self._lock:
            page = self._ensure_page()
            deadline = time.monotonic() + self.response_timeout_s
            poll_s = 1.0

            if not page.evaluate(FILL_JS, text):
                raise ChatError("composer not found (site markup changed?)")
            try:
                page.wait_for_selector(
                    'button[aria-label="Send message"]:not([disabled])',
                    timeout=self.request_timeout_s * 1000,
                )
                page.click('button[aria-label="Send message"]')
            except Exception as exc:
                raise ChatError(f"could not submit message: {exc}") from exc
            logger.info("submitted; waiting for reply")

            # 1. Generation starts ("Stop generating" appears). Fast replies
            # may complete before we observe it -- tolerated below.
            saw_generating = False
            while time.monotonic() < deadline:
                state = page.evaluate(STATE_JS)
                if state.get("stopVisible"):
                    saw_generating = True
                    break
                if state.get("sendReady") and (page.evaluate(LAST_REPLY_JS) or "").strip():
                    return self._finish(page)
                time.sleep(poll_s)

            # 2. Completion: Stop gone AND composer enabled again.
            while time.monotonic() < deadline:
                state = page.evaluate(STATE_JS)
                if saw_generating and not state.get("stopVisible") and state.get("composerReady"):
                    return self._finish(page)
                if not saw_generating and state.get("sendReady"):
                    return self._finish(page)
                time.sleep(poll_s)
            raise ChatError(
                f"reply not completed within {self.response_timeout_s}s"
            )

    def _finish(self, page) -> str:
        reply = (page.evaluate(LAST_REPLY_JS) or "").strip()
        if not reply:
            raise ChatError("empty reply extracted (site markup changed?)")
        logger.info("reply completed (%d chars)", len(reply))
        return reply
