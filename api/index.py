"""Vercel Python Function entrypoint for the existing FastAPI application.

Note: this adapter only exposes the ASGI app. The application still depends on
Playwright/Chromium and background threads; those behaviors must be validated
on Vercel before using this endpoint with live phone calls.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402,F401
