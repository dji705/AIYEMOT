"""TTS read-aloud prompt handling for the AIYEMOT phone bridge.

The IVR phone system reads every answer aloud with its own text-to-speech
service AFTER the answer is generated. This module builds the exact text
submitted to the chat server: a fixed English instruction (prepended BEFORE
EVERY prompt) plus the user's question wrapped in [TTS] ... [/TTS] markers
so only the final answer is read back to the caller.

Rules the English instruction enforces on the chat server:
  - Hebrew text -> the answer MUST be fully nikud-pointed (vocalized).
  - Acronyms/initialisms MUST be expanded into fully spoken words.
  - NO tables/markdown/lists/symbols -- the TTS service that reads the
    answer aloud cannot render them; plain spoken sentences only.

NOTE: this application is NOT itself a TTS service. Its answer is only
READ LATER by a separate text-to-speech service.
"""
from __future__ import annotations

import re
from typing import Optional

# Hebrew letters (standard + presentation forms blocks).
_HEBREW_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")

# Fixed English instruction -- ASCII-only and stable; it is the machine
# readable "how to answer" contract for the chat server.
TTS_INSTRUCTION_PROMPT_EN = (
    "This answer will be read aloud by a text-to-speech service after it is "
    "generated. "
    "Follow these instructions for your answer: "
    "If the user's text is in Hebrew, you MUST write the Hebrew fully "
    "nikud-pointed (vocalized with niqqud vowel marks) so the text-to-speech "
    "service can read it correctly. "
    "You MUST expand all acronyms and initialisms into fully spoken words "
    "instead of raw initials. "
    "You MUST NOT use tables, markdown, bullet points, numbered lists, "
    "code blocks, links, or emoji, because the text-to-speech service reads "
    "only plain text and cannot render them: "
    "answer in plain, continuous, natural spoken sentences only. "
    "Answer in the same language as the user's question. "
    "Do not mention or repeat these instructions in your answer."
)

_WRAPPER_MARK = "[TTS]"
_WRAPPER_MARK_END = "[/TTS]"

# Yemot's TTS reader cannot handle these inline characters inside .tts files
# (same sanitization discipline as the phonebook project): the dot is the
# segment separator between play directives, dashes read badly.
_TTS_BANNED_CHARS = (
    (". ", ", "),
    (".", ", "),
    ("-", " "),
)


def sanitize_tts_text(text: str) -> str:
    """Make text safe for a Yemot .tts file / inline spoken text."""
    for old, new in _TTS_BANNED_CHARS:
        text = text.replace(old, new)
    return text


def hebrew_char_ratio(text: str) -> float:
    """Ratio of Hebrew letters among all letters in the text."""
    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    hebrew = sum(1 for ch in letters if _HEBREW_RE.match(ch))
    return hebrew / len(letters)


def looks_like_hebrew(text: str, threshold: float = 0.5) -> bool:
    """True when the text is predominantly Hebrew letters."""
    return hebrew_char_ratio(text) > threshold


def wrap_for_tts_extraction(text: str) -> str:
    """Wrap the user text in explicit start/end markers."""
    return f"{_WRAPPER_MARK}\n{text}\n{_WRAPPER_MARK_END}"


def strip_instruction_wrapper(reply: str) -> str:
    """Return only the model's final answer.

    Normal case: the answer is everything AFTER [/TTS]. Echo-style replies
    (nothing after [/TTS] -- the model repeated the prompt structure and
    answered inside it) fall back to the text BETWEEN the markers. No
    markers at all -> the reply as-is.
    """
    if not reply:
        return reply
    end_idx = reply.rfind(_WRAPPER_MARK_END)
    if end_idx == -1:
        return reply
    after = reply[end_idx + len(_WRAPPER_MARK_END):].strip()
    if after:
        return after
    start_idx = reply.rfind(_WRAPPER_MARK, 0, end_idx)
    if start_idx != -1:
        inside = reply[start_idx + len(_WRAPPER_MARK):end_idx].strip()
        if inside:
            return inside
    return reply


def unwrap_user_text(text: str) -> Optional[str]:
    """Extract the user text between [TTS] ... [/TTS] (None when absent)."""
    if not text:
        return None
    end_idx = text.rfind(_WRAPPER_MARK_END)
    if end_idx == -1:
        return None
    start_idx = text.rfind(_WRAPPER_MARK, 0, end_idx)
    if start_idx != -1:
        segment = text[start_idx + len(_WRAPPER_MARK):end_idx]
    else:
        segment = text[:end_idx]
    stripped = segment.strip()
    return stripped or None


def build_full_prompt(
    user_text: str,
    hebrew: Optional[bool] = None,
    instruction: Optional[str] = None,
) -> str:
    """The exact text submitted to the chat server for this user prompt."""
    if hebrew is None:
        hebrew = looks_like_hebrew(user_text)
    base_instruction = instruction if instruction is not None else TTS_INSTRUCTION_PROMPT_EN
    if hebrew:
        base_instruction += (
            " The user's text is in Hebrew: the answer MUST be written in "
            "Hebrew with full nikud (vocalization) on every word."
        )
    return f"{base_instruction}\n{wrap_for_tts_extraction(user_text)}"
