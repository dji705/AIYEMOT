"""Shared test fixtures for the AIYEMOT suite.

Installs lightweight stubs for fastapi/httpx/pydub/speech_recognition only
when the real packages aren't importable, loads app.py by path, and wires
every YHM REST helper plus the chat bridge to in-memory fakes.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_TESTS_DIR), str(_TESTS_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


def _install_stubs() -> dict:
    """Ensure app.py's imports resolve (mirrors FIXEDVER2's yhm_stubs)."""
    status: dict[str, str] = {}

    fastapi = _try_import("fastapi")
    if fastapi is None:
        fastapi = types.ModuleType("fastapi")

        class FastAPI:
            def __init__(self, *args, **kwargs):
                pass

            def api_route(self, *args, **kwargs):
                def deco(fn):
                    return fn
                return deco

            def get(self, *args, **kwargs):
                def deco(fn):
                    return fn
                return deco

        fastapi.FastAPI = FastAPI
        fastapi.Request = object
        sys.modules["fastapi"] = fastapi
        status["fastapi"] = "stub"

    responses_mod = _try_import("fastapi.responses")
    if responses_mod is None or not hasattr(responses_mod, "PlainTextResponse"):
        responses_mod = types.ModuleType("fastapi.responses")

        class PlainTextResponse:
            def __init__(self, content="", *args, **kwargs):
                self.body = content
                self.content = content

        responses_mod.PlainTextResponse = PlainTextResponse
        fastapi.responses = responses_mod
        sys.modules["fastapi.responses"] = responses_mod
        status["fastapi.responses"] = "stub" if status.get("fastapi") == "stub" else "partial"
    else:
        status["fastapi.responses"] = "real"

    if _try_import("pydub") is None:
        pydub = types.ModuleType("pydub")
        pydub.AudioSegment = type("AudioSegment", (), {})
        sys.modules["pydub"] = pydub
        status["pydub"] = "stub"

    if _try_import("speech_recognition") is None:
        sr = types.ModuleType("speech_recognition")
        sr.Recognizer = type("Recognizer", (), {})
        sr.AudioFile = type("AudioFile", (), {})
        sr.UnknownValueError = type("UnknownValueError", (Exception,), {})
        sr.RequestError = type("RequestError", (Exception,), {})
        sys.modules["speech_recognition"] = sr
        status["speech_recognition"] = "stub"

    if _try_import("httpx") is None:
        httpx = types.ModuleType("httpx")
        httpx.AsyncClient = type("AsyncClient", (), {})
        sys.modules["httpx"] = httpx
        status["httpx"] = "stub"

    return status


STUB_STATUS = _install_stubs()


class FakeBridge:
    """In-memory stand-in for chat_bridge.PlaywrightChatBridge."""

    def __init__(self):
        self.submitted: list[str] = []
        self.reset_count = 0
        self.target_url = "https://chatgpt.test/"
        self.next_reply = "תְּשׁוּבָה מְנֻקַּדֶת"
        self.fail_next = False

    def send_and_wait(self, text: str) -> str:
        self.submitted.append(text)
        if self.fail_next:
            self.fail_next = False
            from chat_bridge import ChatError

            raise ChatError("simulated chat failure")
        return f"INSTRUCTION HEADER\n[TTS]\n{self.next_reply}\n[/TTS]"

    def reset_conversation(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        pass


class FakeYHM:
    """In-memory replacement for the YHM REST surface app.py uses."""

    def __init__(self):
        self.files: dict[str, str] = {}
        self.wavs: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.write_log: list[tuple[str, str]] = []

    async def read_text_file(self, path: str, token: str):
        return self.files.get(path)

    async def write_text_file(self, path: str, contents: str, token: str) -> bool:
        self.files[path] = contents
        self.write_log.append((path, contents))
        return True

    async def list_dir(self, path: str, token: str):
        prefix = path if path.endswith("/") else path + "/"
        out = []
        for name, data in self.wavs.items():
            full = prefix + name
            if full in self.wavs or name in self.wavs:
                out.append({"name": name, "size": len(data)})
        # simple contract: wavs are keyed by bare name within this dir
        return [{"name": n, "size": len(d)} for n, d in self.wavs.items()] if path.endswith("/") else []

    async def download_file(self, path: str, token: str):
        name = path.rsplit("/", 1)[-1]
        return self.wavs.get(name)

    async def delete_files(self, paths, token: str) -> bool:
        for p in list(paths):
            self.deleted.append(p)
            self.files.pop(p, None)
            self.wavs.pop(p.rsplit("/", 1)[-1], None)
        return True

    def put(self, path: str, contents: str) -> None:
        self.files[path] = contents

    def wrote(self, suffix: str):
        hits = [(p, c) for p, c in self.write_log if p.endswith(suffix)]
        return hits[-1] if hits else None


def _load_app():
    module_path = _TESTS_DIR.parent / "app.py"
    spec = importlib.util.spec_from_file_location("aiyemot_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aiyemot_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def app_mod():
    return _load_app()


@pytest.fixture()
def env(app_mod, monkeypatch):
    """Fresh fakes wired into the app module for one test."""
    fake_yhm = FakeYHM()
    fake_bridge = FakeBridge()

    monkeypatch.setattr(app_mod, "yhm_read_text_file", fake_yhm.read_text_file)
    monkeypatch.setattr(app_mod, "yhm_write_text_file", fake_yhm.write_text_file)
    monkeypatch.setattr(app_mod, "yhm_list_dir", fake_yhm.list_dir)
    monkeypatch.setattr(app_mod, "yhm_download_file", fake_yhm.download_file)
    monkeypatch.setattr(app_mod, "yhm_delete_files", fake_yhm.delete_files)
    monkeypatch.setattr(app_mod, "chat", fake_bridge)
    monkeypatch.setattr(app_mod, "HAS_STT", True)

    # Make capture_latest_recording use the fake WAV store: seed a fake
    # "recording" by replacing STT with a fixed transcription per test.
    return {"yhm": fake_yhm, "bridge": fake_bridge, "app": app_mod}


@pytest.fixture()
def stt_text(app_mod, monkeypatch):
    """Control what capture_latest_recording 'hears' (skips WAV pipeline)."""

    def _set(text_or_none: str | None):
        async def _fake(ext: str, token: str):
            return text_or_none

        monkeypatch.setattr(app_mod, "capture_latest_recording", _fake)

    return _set


def wait_for_threads(app_mod, timeout_s: float = 5.0) -> None:
    """Join all in-flight chat job threads (started by the endpoint)."""
    for t in threading.enumerate():
        if t.name.startswith("chat-") and t is not threading.current_thread():
            t.join(timeout=timeout_s)
