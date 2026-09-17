# ═══════════════════════════════════════════════════════════════════════════
# app.py — AIYEMOT: Voice ChatGPT bridge for Yemot HaMashiach (YHM)
# ═══════════════════════════════════════════════════════════════════════════
#
# WHAT THIS IS
# ------------
# A `type=api` webhook (FastAPI, deployed e.g. on Render) implementing a
# conversational voice loop for Yemot HaMashiach:
#
#   1. The caller enters the API extension -> hears "מה שאלתך?" and is
#      routed to the recording sub-extension /1.
#   2. The caller records a question; the recording sub-extension sends
#      them straight back up to the API extension (record_end_goto).
#   3. The webhook transcribes the newest numeric WAV (Google STT, he-IL)
#      and deletes the WAV immediately (the proven bookchanger discipline).
#   4. If the transcription is exactly a logout word (LOG OUT / LOGOUT /
#      EXIT, case-insensitive), the ChatGPT conversation is reset, a short
#      confirmation plays, and the loop continues within the same call.
#   5. Otherwise the question is submitted to chatgpt.com's anonymous chat
#      (embedded Playwright bridge, chat_bridge.py) WITH the English
#      read-aloud instruction prompt (tts_prompt.py) prepended BEFORE
#      every submission. Generation runs in a background thread (a reply
#      takes 10-60s; YHM's webhook timeout is far shorter), so the webhook
#      answers instantly: it writes a "ברגע סבלנות" placeholder .tts into
#      sub-extension /2 and routes the caller there.
#   6. Sub-extension /2 is type=play_and_return: it plays the answer file
#      (overwritten by the real answer as soon as it is ready) and
#      automatically returns up to the API extension. The loop continues:
#      greeting -> record in /1 -> answer in /2 -> greeting ... as long as
#      the same phone call is alive. A NEW call (new ApiCallId) restarts
#      the IVR loop but KEEPS the ChatGPT conversation; only a spoken
#      logout word opens a fresh ChatGPT session.
#
# ANSWER FILE: ivr2:/{ext}/2/000.tts -- it must live INSIDE sub-extension
# /2 because play_and_return plays that folder's own files. Single active
# caller assumed (personal line): two simultaneous callers would share the
# same 000.tts.
#
# QUESTION QUEUE: if a new question is recorded while the previous answer
# is still generating, it is stored as QUESTION_{phone}.tts and the
# running job chains to it when it finishes -- a question is never lost,
# only delayed.
#
# STATE MAP (state_{phone}.tts on the YHM server: ApiCallId / state -- the
# exact durable-state convention bookchanger.py uses; nothing is kept in
# memory or on local disk, so this works on stateless free hosting):
#   1  Say "מה שאלתך?" -> route to recording /1 (state already 2 for the
#      recording's return trip)
#   2  Recording landed: STT -> logout? (reset + ack -> 2, route /1)
#      else: placeholder into /2/000.tts -> background chat -> route /2 (->3)
#   3  Returned from /2 (answer heard): greeting -> route /1 (->2)
# ═══════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import threading

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

from chat_bridge import ChatError, PlaywrightChatBridge
from tts_prompt import (
    build_full_prompt,
    looks_like_hebrew,
    sanitize_tts_text,
    strip_instruction_wrapper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aiyemot.app")

app = FastAPI(
    title="AIYEMOT — Voice ChatGPT bridge for Yemot HaMashiach",
    description="שאל בקול, קבל תשובה מנוקדת מוקראת בטלפון",
    version="1.0.0",
)

# ─── Configuration ───────────────────────────────────────────────────────────

_yhm_env = os.environ.get("YHM_API_BASE", "").strip()
if not _yhm_env:
    YHM_API_BASE = "https://www.call2all.co.il/ym/api"
else:
    if not _yhm_env.startswith(("http://", "https://")):
        _yhm_env = "https://" + _yhm_env
    YHM_API_BASE = _yhm_env.rstrip("/")

# The YHM API token arrives from ext.ini's api_add_0=token=... on every
# webhook call (never stored server-side). YHM_API_TOKEN can pin it for
# local testing only.
DEFAULT_TOKEN = os.environ.get("YHM_API_TOKEN", "").strip()

# Spoken logout words (EXACT match, case-insensitive) that reset the ChatGPT
# conversation without hanging up.
LOGOUT_WORDS = {"log out", "logout", "exit"}

REQUEST_TIMEOUT_S = int(os.environ.get("CHAT_REQUEST_TIMEOUT_S", "30"))
RESPONSE_TIMEOUT_S = int(os.environ.get("CHAT_RESPONSE_TIMEOUT_S", "180"))
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "he-IL")

GREETING = "מה שאלתך?"
WAIT_TEXT = "ברגע סבלנות, התשובה בעיבוד, עמוד על הקו"
LOGOUT_ACK = "נפתח סשן חדש. " + GREETING
UNCLEAR_TEXT = "דיבור לא ברור, אנא חזרו על השאלה"
BUSY_TEXT = "רגע, מסיימים לעבד את השאלה הקודמת, עמודו על הקו"
ERROR_TEXT = "מתנצלים, לא הצלחנו להשיג תשובה, אנא נסו שוב"

# The answer file INSIDE sub-extension /2 (played by its play_and_return
# module). 000.tts sorts first and matches YHM's playback naming.
ANSWER_BASENAME = "000"

# One shared, thread-safe bridge for all calls (one browser, one page;
# submissions are serialized by the bridge's internal lock).
chat = PlaywrightChatBridge(
    request_timeout_s=REQUEST_TIMEOUT_S,
    response_timeout_s=RESPONSE_TIMEOUT_S,
)

# Phones with a chat job currently generating.
_inflight: set[str] = set()
_inflight_lock = threading.Lock()

# STT imports (same pipeline as bookchanger.py; tolerant so tests can import
# the module without the audio stack installed).
try:
    from pydub import AudioSegment
    import speech_recognition as sr

    HAS_STT = True
except Exception:  # pragma: no cover - tests stub these
    AudioSegment = None
    sr = None
    HAS_STT = False


# ─── YHM REST helpers (httpx, same shapes as bookchanger.py) ────────────────

def get_yhm_path(ext: str, file_name: str) -> str:
    clean_ext = ext.strip("/")
    if clean_ext:
        return f"ivr2:/{clean_ext}/{file_name}"
    return f"ivr2:/{file_name}"


async def yhm_read_text_file(path: str, token: str) -> str | None:
    url = f"{YHM_API_BASE}/GetTextFile"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params={"token": token, "what": path})
            if resp.status_code != 200:
                logger.warning("GetTextFile HTTP %d for %s", resp.status_code, path)
                return None
            payload = resp.json()
            if payload.get("responseStatus") == "OK":
                return payload.get("contents")
            logger.warning("GetTextFile failed for %s: %s", path, payload.get("message", ""))
            return None
        except Exception as exc:
            logger.error("GetTextFile error %s: %s", path, exc)
            return None


async def yhm_write_text_file(path: str, contents: str, token: str) -> bool:
    url = f"{YHM_API_BASE}/UploadTextFile"
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt, method in enumerate(["POST", "GET"], start=1):
            try:
                params = {"token": token, "what": path, "contents": contents}
                if method == "POST":
                    resp = await client.post(url, data=params)
                else:
                    resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    logger.warning("UploadTextFile try %d/%s: HTTP %d for %s",
                                   attempt, method, resp.status_code, path)
                    continue
                payload = resp.json()
                if payload.get("responseStatus") == "OK":
                    return True
                logger.error("UploadTextFile (%s) failed for %s: %s",
                             method, path, payload.get("message", ""))
            except Exception as exc:
                logger.error("UploadTextFile (%s) try %d error for %s: %s",
                             method, attempt, path, exc)
    return False


async def yhm_list_dir(path: str, token: str) -> list[dict] | None:
    url = f"{YHM_API_BASE}/GetIVR2Dir"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params={"token": token, "path": path})
            if resp.status_code != 200:
                logger.error("GetIVR2Dir HTTP %d for %s", resp.status_code, path)
                return None
            payload = resp.json()
            if payload.get("responseStatus") == "OK":
                return payload.get("files", [])
            logger.warning("GetIVR2Dir failed for %s: %s", path, payload.get("message", ""))
            return None
        except Exception as exc:
            logger.error("GetIVR2Dir error %s: %s", path, exc)
            return None


async def yhm_download_file(path: str, token: str) -> bytes | None:
    url = f"{YHM_API_BASE}/DownloadFile"
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, params={"token": token, "path": path})
            if resp.status_code == 200:
                body = resp.content
                if body.strip().startswith(b'{"'):
                    try:
                        data = json.loads(body.decode("utf-8"))
                        if data.get("responseStatus") not in (None, "OK"):
                            logger.error("YHM API error downloading %s: %s",
                                         path, data.get("message", ""))
                            return None
                    except Exception:
                        pass
                return body
            logger.error("DownloadFile HTTP %d for %s", resp.status_code, path)
            return None
        except Exception as exc:
            logger.error("DownloadFile error %s: %s", path, exc)
            return None


async def yhm_delete_files(paths: list[str], token: str) -> bool:
    if not paths:
        return True
    url = f"{YHM_API_BASE}/FileAction"
    params: dict = {"token": token, "action": "delete"}
    for i, path in enumerate(paths):
        params[f"what{i}"] = path
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("responseStatus") == "OK":
                    return True
                logger.error("FileAction delete failed: %s", payload.get("message", ""))
            else:
                logger.error("FileAction delete HTTP %d", resp.status_code)
        except Exception as exc:
            logger.error("FileAction delete error: %s", exc)
    return False


# ─── Recording capture + STT (bookchanger discipline) ───────────────────────

def extract_latest_sequential_wav(files: list[dict]) -> dict | None:
    """The highest-numbered NNN.wav (newest recording) in the folder."""
    wav_files = []
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith(".wav"):
            base = name[:-4]
            if base.isdigit():
                wav_files.append((int(base), f))
    if not wav_files:
        return None
    wav_files.sort(key=lambda x: x[0])
    return wav_files[-1][1]


def process_audio_stt(audio_bytes: bytes, file_name: str) -> str:
    """Google STT (he-IL by default) with 1s silence padding."""
    if not HAS_STT:
        logger.error("STT dependencies not installed (pydub/speech_recognition).")
        return ""
    recognizer = sr.Recognizer()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_in:
        temp_in.write(audio_bytes)
        temp_in.flush()
        try:
            try:
                audio = AudioSegment.from_file(temp_in.name, format="wav")
            except Exception:
                audio = AudioSegment.from_file(temp_in.name)
            silence = AudioSegment.silent(duration=1000)
            processed = silence + audio + silence
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_out:
                processed.export(temp_out.name, format="wav")
                with sr.AudioFile(temp_out.name) as source:
                    data = recognizer.record(source)
            text = recognizer.recognize_google(data, language=STT_LANGUAGE)
            logger.info("STT recognized: '%s'", text)
            return text
        except sr.UnknownValueError:
            logger.warning("STT: no clear speech in the recording.")
            return ""
        except sr.RequestError as req_err:
            logger.error("STT service error: %s", req_err)
            return ""
        except Exception as exc:
            logger.error("STT general error: %s", exc)
            return ""


async def capture_latest_recording(ext: str, token: str) -> str | None:
    """Download the newest numeric WAV from `ext`, delete it immediately and
    unconditionally, then STT it. Returns the recognized text or None."""
    files = await yhm_list_dir(get_yhm_path(ext, ""), token)
    latest = extract_latest_sequential_wav(files) if files else None
    if not latest:
        logger.warning("no numeric WAV found in %s", ext)
        return None

    filename = latest["name"]
    audio_bytes = await yhm_download_file(get_yhm_path(ext, filename), token)

    # Immediate, unconditional deletion of the numeric WAVs -- the exact
    # Bug #1 discipline from bookchanger.py (a stale recording must never
    # be re-processed as a "new" question on the next loop turn).
    wav_paths = [
        get_yhm_path(ext, f["name"])
        for f in (files or [])
        if f.get("name", "").lower().endswith(".wav")
        and f.get("name", "")[:-4].isdigit()
    ]
    await yhm_delete_files(wav_paths, token)

    if not audio_bytes or len(audio_bytes) < 1000:
        logger.warning("recording missing/too short (%s)", filename)
        return None
    return process_audio_stt(audio_bytes, filename)


# ─── Per-call state (durable, on the YHM server) ────────────────────────────

async def save_caller_state(ext: str, phone: str, api_call_id: str,
                            state: int, token: str) -> bool:
    path = get_yhm_path(ext, f"state_{phone}.tts")
    ok = await yhm_write_text_file(path, f"{api_call_id}\n{state}\n", token)
    if not ok:
        logger.critical("state save FAILED phone=%s ext=%s state=%d", phone, ext, state)
    return ok


def _safe_int(raw: str) -> int | None:
    try:
        return int(raw)
    except ValueError:
        return None


async def read_caller_state(ext: str, phone: str, token: str) -> tuple[str, int | None]:
    """Returns (stored_call_id, stored_state). ('', None) when absent."""
    content = await yhm_read_text_file(get_yhm_path(ext, f"state_{phone}.tts"), token)
    if not content:
        return "", None
    lines = content.splitlines()
    if len(lines) >= 2:
        return lines[0].strip(), _safe_int(lines[1].strip())
    return "", None


def is_logout_word(text: str) -> bool:
    """True when the transcription is EXACTLY one of the logout words."""
    return text.strip().lower() in LOGOUT_WORDS


# ─── Background chat generation (with question chaining) ────────────────────

def _run_chat_job(phone: str, ext: str, token: str, question: str) -> None:
    """Thread body: submit the TTS-prompted question through the bridge and
    write the answer into /2/000.tts (overwriting the WAIT placeholder).
    Never leaves a stuck placeholder: on failure it writes ERROR_TEXT.
    When done, chains to a question queued meanwhile (QUESTION file)."""
    answer_path = get_yhm_path(f"{ext}/2", f"{ANSWER_BASENAME}.tts")
    full_prompt = build_full_prompt(question)
    logger.info("chat job start phone=%s hebrew=%s prompt_chars=%d",
                phone, looks_like_hebrew(question), len(full_prompt))
    try:
        raw_reply = chat.send_and_wait(full_prompt)
        reply = strip_instruction_wrapper(raw_reply) or ERROR_TEXT
        logger.info("chat job done phone=%s reply_chars=%d", phone, len(reply))
    except ChatError as exc:
        logger.error("chat job failed phone=%s: %s", phone, exc)
        reply = ERROR_TEXT
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat job crashed phone=%s: %s", phone, exc)
        reply = ERROR_TEXT

    import asyncio

    async def _write_answer() -> str | None:
        ok = await yhm_write_text_file(answer_path, sanitize_tts_text(reply), token)
        if not ok:
            logger.error("failed writing answer file for %s (placeholder kept)", phone)
            return None
        # Chain: was a newer question recorded while we were generating?
        queued_path = get_yhm_path(ext, f"QUESTION_{phone}.tts")
        queued = await yhm_read_text_file(queued_path, token)
        if queued and queued.strip():
            await yhm_delete_files([queued_path], token)
            return queued.strip()
        return None

    try:
        next_question = asyncio.run(_write_answer())
    except Exception as exc:  # pragma: no cover
        logger.error("answer write crashed for %s: %s", phone, exc)
        next_question = None
    finally:
        with _inflight_lock:
            _inflight.discard(phone)

    if next_question:
        logger.info("chaining queued question for phone=%s", phone)
        start_chat_job(phone, ext, token, next_question)


def is_job_running(phone: str) -> bool:
    with _inflight_lock:
        return phone in _inflight


def start_chat_job(phone: str, ext: str, token: str, question: str) -> bool:
    """Start background generation for this caller; False if one is already
    running (the caller should queue via the QUESTION file instead)."""
    with _inflight_lock:
        if phone in _inflight:
            return False
        _inflight.add(phone)
    thread = threading.Thread(
        target=_run_chat_job, args=(phone, ext, token, question),
        name=f"chat-{phone}", daemon=True,
    )
    thread.start()
    return True


# ─── Response helpers ───────────────────────────────────────────────────────

def say_and_go(ext: str, text: str, routing_path: str) -> PlainTextResponse:
    """Play `text` as TTS, then route the caller to `routing_path`."""
    return PlainTextResponse(
        f"id_list_message=t-{sanitize_tts_text(text)}&go_to_folder={routing_path}"
    )


# ─── The Main Endpoint ──────────────────────────────────────────────────────

@app.api_route("/ivr", methods=["GET", "POST"])
async def yemot_ivr(request: Request):
    query_params = dict(request.query_params)
    form_params: dict = {}
    if request.method == "POST":
        try:
            form_params = dict(await request.form())
        except Exception as exc:
            logger.error("form parsing failed (python-multipart installed?): %s", exc)
    params = {**query_params, **form_params}

    if params.get("hangup", "").strip().lower() == "yes":
        logger.info("hangup notification for call %s -- ignoring.",
                    params.get("ApiCallId", ""))
        return PlainTextResponse("")

    api_phone = params.get("ApiPhone", "").strip()
    api_call_id = params.get("ApiCallId", "").strip()
    api_extension = params.get("ApiExtension", "").strip()

    token = (params.get("token") or DEFAULT_TOKEN).strip()
    if not token:
        logger.error("no YHM token (ext.ini api_add or YHM_API_TOKEN env)")
        return PlainTextResponse("id_list_message=t-שגיאה בטוקן")
    if not api_phone:
        return PlainTextResponse("id_list_message=t-שגיאה במספר טלפון")
    if not api_call_id:
        return PlainTextResponse("id_list_message=t-שגיאה במזהה שיחה")

    safe_phone = "".join(ch for ch in api_phone if ch.isalnum())
    ext = api_extension.strip("/")
    record_path = f"/{ext}/1" if ext else "/1"
    answer_path_dir = f"{ext}/2" if ext else "2"
    answer_file_path = get_yhm_path(answer_path_dir, f"{ANSWER_BASENAME}.tts")
    question_file_path = get_yhm_path(ext, f"QUESTION_{safe_phone}.tts")

    stored_call_id, stored_state = await read_caller_state(ext, safe_phone, token)

    # A NEW call (different ApiCallId) restarts the IVR loop from the
    # greeting and clears a stale answer file in /2, but KEEPS the ChatGPT
    # conversation -- only a spoken logout word resets that.
    if stored_call_id and stored_call_id != api_call_id:
        logger.info("new call for phone=%s (old=%s new=%s) -- loop restarts, "
                    "chat conversation kept", safe_phone, stored_call_id, api_call_id)
        current_state = 1
        await yhm_delete_files([answer_file_path], token)
    else:
        current_state = stored_state or 1

    logger.info("--- request --- call=%s phone=%s ext=%s state=%s",
               api_call_id, safe_phone, ext, current_state)

    # ═══ State 1: greeting -> route to the recording sub-extension ════════
    if current_state == 1:
        saved = await save_caller_state(ext, safe_phone, api_call_id, 2, token)
        if not saved:
            return PlainTextResponse("id_list_message=t-שגיאה פנימית במערכת")
        return say_and_go(ext, GREETING, record_path)

    # ═══ State 2: a recording landed -> STT -> logout? / start chat ═══════
    if current_state == 2:
        text = await capture_latest_recording(ext, token)
        if not text:
            # Nothing usable: stay in the loop (state file already says 2).
            return say_and_go(ext, UNCLEAR_TEXT, record_path)

        if is_logout_word(text):
            logger.info("logout word from phone=%s -- resetting chat session", safe_phone)
            # Background reset: reloading the chat page can take several
            # seconds -- far too long to block the webhook response. The
            # caller's NEXT question (recorded seconds later) will already
            # land in the fresh conversation.
            threading.Thread(
                target=chat.reset_conversation, name="chat-reset", daemon=True
            ).start()
            await save_caller_state(ext, safe_phone, api_call_id, 2, token)
            return say_and_go(ext, LOGOUT_ACK, record_path)

        if is_job_running(safe_phone):
            # A generation for this caller is already in flight: queue the
            # question (the running job chains to it when it finishes) and
            # send the caller straight back to recording -- the /2 file is
            # NOT touched (the pending answer must survive).
            await yhm_write_text_file(question_file_path, text, token)
            logger.info("question from phone=%s queued (job in flight)", safe_phone)
            return say_and_go(ext, BUSY_TEXT, record_path)

        # Write the WAIT placeholder BEFORE starting the job: the thread can
        # only overwrite it later, so a fast answer can never be clobbered
        # by the placeholder. Sub-extension /2 now always has a .tts to play.
        placeholder_ok = await yhm_write_text_file(
            answer_file_path, sanitize_tts_text(WAIT_TEXT), token
        )
        if not placeholder_ok:
            logger.error("placeholder write failed phone=%s", safe_phone)

        if not start_chat_job(safe_phone, ext, token, text):
            # Rare race (another webhook started a job meanwhile): fall back
            # to queueing; the placeholder stays until that job chains.
            await yhm_write_text_file(question_file_path, text, token)
            return say_and_go(ext, BUSY_TEXT, record_path)

        logger.info("question from phone=%s (%d chars) -> chat job started",
                    safe_phone, len(text))

        await save_caller_state(ext, safe_phone, api_call_id, 3, token)
        return PlainTextResponse(f"go_to_folder=/{answer_path_dir}")

    # ═══ State 3: caller returned from /2 (answer heard) -> next question ═
    if current_state == 3:
        saved = await save_caller_state(ext, safe_phone, api_call_id, 2, token)
        if not saved:
            return PlainTextResponse("id_list_message=t-שגיאה פנימית במערכת")
        return say_and_go(ext, GREETING, record_path)

    # ═══ Unknown state -> restart the loop cleanly ════════════════════════
    logger.warning("unknown state %s for phone=%s -- restarting loop",
                   current_state, safe_phone)
    await save_caller_state(ext, safe_phone, api_call_id, 2, token)
    return say_and_go(ext, GREETING, record_path)


# ─── Health Check ───────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "engine": "AIYEMOT v1.0.0",
        "chat_target": chat.target_url,
        "stt_ready": HAS_STT,
    }


if __name__ == "__main__":  # pragma: no cover - local run convenience
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
