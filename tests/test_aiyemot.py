"""AIYEMOT test suite (fully offline).

Covers:
  1. tts_prompt: prompt building, Hebrew detection, wrapper strip, sanitize.
  2. Logout words: exact matching (LOG OUT / LOGOUT / EXIT, case-insensitive)
     and non-matching questions.
  3. The state machine end-to-end with fakes: greeting -> record return ->
     chat job -> placeholder -> answer written into /2/000.tts -> return ->
     greeting again. New call resets the loop but keeps the conversation.
  4. Failure paths: STT empty, chat failure (ERROR_TEXT), job chaining.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
import time
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_TESTS_DIR), str(_TESTS_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import app as app_mod  # loaded by conftest under pytest

    _UNDER_PYTEST = True
except ImportError:
    import conftest  # noqa: F401  (installs stubs)

    app_mod = conftest._load_app()
    _UNDER_PYTEST = False

from conftest import FakeBridge, FakeYHM, wait_for_threads  # noqa: E402

EXT = "100"
CALLER = "0500000000"
STATE = f"ivr2:/{EXT}/state_{CALLER}.tts"
ANSWER = f"ivr2:/{EXT}/2/000.tts"
QUESTION_FILE = f"ivr2:/{EXT}/QUESTION_{CALLER}.tts"


def run(coro):
    return asyncio.run(coro)


class FakeRequest:
    def __init__(self, params: dict):
        self.method = "GET"
        self.query_params = params

    async def form(self):
        return {}


def base_params(**over) -> dict:
    params = {
        "ApiPhone": CALLER,
        "ApiCallId": "CALL1",
        "ApiExtension": EXT,
        "token": "TOK",
    }
    params.update(over)
    return params


def call(params: dict):
    return asyncio.run(app_mod.yemot_ivr(FakeRequest(params)))


def body_of(resp) -> str:
    """Normalize a PlainTextResponse body to str (real fastapi returns bytes)."""
    body = getattr(resp, "body", None)
    if body is None:
        body = resp.content
    return body.decode("utf-8") if isinstance(body, bytes) else body


def seed_state(fake_yhm: FakeYHM, state: int, call_id: str = "CALL1"):
    fake_yhm.put(STATE, f"{call_id}\n{state}\n")


def wire_env(fake_yhm: FakeYHM, fake_bridge: FakeBridge):
    app_mod.yhm_read_text_file = fake_yhm.read_text_file
    app_mod.yhm_write_text_file = fake_yhm.write_text_file
    app_mod.yhm_list_dir = fake_yhm.list_dir
    app_mod.yhm_download_file = fake_yhm.download_file
    app_mod.yhm_delete_files = fake_yhm.delete_files
    app_mod.chat = fake_bridge
    app_mod.HAS_STT = True


# ═══ 1. tts_prompt ══════════════════════════════════════════════════════════

def test_full_prompt_starts_with_english_instruction():
    from tts_prompt import TTS_INSTRUCTION_PROMPT_EN, build_full_prompt

    prompt = build_full_prompt("מה השעה?")
    assert prompt.startswith(TTS_INSTRUCTION_PROMPT_EN)
    assert "מה השעה?" in prompt


def test_prompt_hebrew_gets_nikud_emphasis():
    from tts_prompt import build_full_prompt

    assert "full nikud" in build_full_prompt("שלום")
    assert "full nikud" not in build_full_prompt("what time is it?")[len(
        "This answer will be read aloud by a text-to-speech service after it is generated."
    ):]


def test_prompt_forbids_tables_and_expands_acronyms():
    from tts_prompt import build_full_prompt

    lowered = build_full_prompt("hello").lower()
    assert "tables" in lowered
    assert "acronyms" in lowered
    assert "will be read aloud by a text-to-speech service" in lowered


def test_strip_wrapper_returns_answer_only():
    from tts_prompt import strip_instruction_wrapper

    reply = "instructions echo\n[/TTS]\nתְּשׁוּבָה"
    assert strip_instruction_wrapper(reply) == "תְּשׁוּבָה"
    assert strip_instruction_wrapper("plain") == "plain"
    assert strip_instruction_wrapper("echo\n[/TTS]") == "echo\n[/TTS]"


def test_sanitize_tts_text():
    from tts_prompt import sanitize_tts_text

    assert sanitize_tts_text("א. ב") == "א, ב"
    assert sanitize_tts_text("א-ב") == "א ב"


def test_is_logout_words():
    assert app_mod.is_logout_word("LOG OUT")
    assert app_mod.is_logout_word("logout")
    assert app_mod.is_logout_word("Exit")
    assert app_mod.is_logout_word("  exit  ")
    assert not app_mod.is_logout_word("please exit")
    assert not app_mod.is_logout_word("logout now")
    assert not app_mod.is_logout_word("יציאה")
    assert not app_mod.is_logout_word("")


# ═══ 2+3. State machine end-to-end ══════════════════════════════════════════

def test_new_call_greets_and_routes_to_record():
    yhm, bridge = FakeYHM(), FakeBridge()
    wire_env(yhm, bridge)
    resp = call(base_params())
    body = body_of(resp)
    assert "מה שאלתך?" in body
    assert "go_to_folder=/100/1" in body
    assert yhm.files[STATE].endswith("\n2\n")


def test_record_return_starts_chat_and_routes_to_answer_ext():
    yhm, bridge = FakeYHM(), FakeBridge()
    wire_env(yhm, bridge)
    seed_state(yhm, 2)

    async def _stt(ext, token):
        return "מה הבירה הפופולרית ביותר בישראל?"

    app_mod.capture_latest_recording = _stt
    try:
        resp = call(base_params())
    finally:
        wait_for_threads(app_mod)

    # The question was submitted WITH the English TTS instruction:
    assert len(bridge.submitted) == 1
    assert "text-to-speech service" in bridge.submitted[0]
    assert "מה הבירה הפופולרית ביותר בישראל?" in bridge.submitted[0]
    # Placeholder was written then overwritten by the real answer in /2:
    assert "ברגע סבלנות" in yhm.wrote("2/000.tts")[1] or ANSWER in yhm.files
    assert yhm.files[ANSWER] == "תְּשׁוּבָה מְנֻקַּדֶת"
    # Caller routed to the answer extension and state advanced to 3:
    assert "go_to_folder=/100/2" in body_of(resp)
    assert yhm.files[STATE].endswith("\n3\n")


def test_return_from_answer_ext_greets_again():
    yhm, bridge = FakeYHM(), FakeBridge()
    wire_env(yhm, bridge)
    seed_state(yhm, 3)
    resp = call(base_params())
    assert "מה שאלתך?" in body_of(resp)
    assert "go_to_folder=/100/1" in body_of(resp)
    assert yhm.files[STATE].endswith("\n2\n")


def test_logout_word_resets_chat_but_continues_call():
    yhm, bridge = FakeYHM(), FakeBridge()
    wire_env(yhm, bridge)
    seed_state(yhm, 2)

    async def _stt(ext, token):
        return "LOGOUT"

    app_mod.capture_latest_recording = _stt
    resp = call(base_params())

    assert bridge.reset_count == 1
    assert bridge.submitted == []  # nothing was sent to the chat
    assert "נפתח סשן חדש" in body_of(resp)
    assert "מה שאלתך?" in body_of(resp)  # continues the loop in the same call
    assert "go_to_folder=/100/1" in body_of(resp)
    assert yhm.files[STATE].endswith("\n2\n")


def test_unclear_recording_repeats_request():
    yhm, bridge = FakeYHM(), FakeBridge()
    wire_env(yhm, bridge)
    seed_state(yhm, 2)

    async def _stt(ext, token):
        return None

    app_mod.capture_latest_recording = _stt
    resp = call(base_params())
    assert "דיבור לא ברור" in body_of(resp)
    assert "go_to_folder=/100/1" in body_of(resp)
    assert bridge.submitted == []


def test_chat_failure_writes_error_text():
    yhm, bridge = FakeYHM(), FakeBridge()
    bridge.fail_next = True
    wire_env(yhm, bridge)
    seed_state(yhm, 2)

    async def _stt(ext, token):
        return "שאלה"

    app_mod.capture_latest_recording = _stt
    call(base_params())
    wait_for_threads(app_mod)
    assert yhm.files[ANSWER] == app_mod.ERROR_TEXT


def test_new_call_deletes_stale_answer_but_keeps_conversation():
    yhm, bridge = FakeYHM(), FakeBridge()
    wire_env(yhm, bridge)
    seed_state(yhm, 3, call_id="OLDCALL")
    yhm.files[ANSWER] = "תשובה ישנה"

    resp = call(base_params(ApiCallId="CALL2"))

    # Loop restarts at the greeting...
    assert "מה שאלתך?" in body_of(resp)
    # ...the stale answer was cleaned up...
    assert ANSWER in yhm.deleted
    # ...and the ChatGPT conversation was NOT reset:
    assert bridge.reset_count == 0
    assert bridge.submitted == []


def test_queued_question_chains_after_current_job():
    yhm, bridge = FakeYHM(), FakeBridge()
    wire_env(yhm, bridge)
    seed_state(yhm, 2)

    # Simulate "a job is already running" for this caller:
    with app_mod._inflight_lock:
        app_mod._inflight.add(CALLER)

    async def _stt(ext, token):
        return "שאלה שנייה"

    app_mod.capture_latest_recording = _stt
    resp = call(base_params())
    assert "מסיימים לעבד" in body_of(resp)  # BUSY_TEXT
    assert yhm.files[QUESTION_FILE] == "שאלה שנייה"
    # The queued question stays until the current job chains to it:
    with app_mod._inflight_lock:
        app_mod._inflight.discard(CALLER)


def test_job_chain_picks_up_queued_question():
    """The running job chains to the QUESTION file after finishing."""
    yhm, bridge = FakeYHM(), FakeBridge()
    wire_env(yhm, bridge)
    yhm.put(QUESTION_FILE, "שאלה שנייה")

    app_mod._run_chat_job(CALLER, EXT, "TOK", "שאלה ראשונה")

    assert len(bridge.submitted) == 2
    assert "שאלה ראשונה" in bridge.submitted[0]
    assert "שאלה שנייה" in bridge.submitted[1]
    assert QUESTION_FILE not in yhm.files  # consumed


# ═══ Zero-install fallback runner ═══════════════════════════════════════════

if __name__ == "__main__":
    if not _UNDER_PYTEST:
        conftest._install_stubs()

    failures = []
    ran = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            ran += 1
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{ran} passed, {len(failures)} failed | stubs: {conftest.STUB_STATUS}")
    sys.exit(1 if failures else 0)
