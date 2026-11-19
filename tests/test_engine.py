"""The one real-engine check. Contract tests keep using the fake transcriber.

Run it after the weights are on disk:

    WHISPER_RUN_ENGINE=1 .venv/bin/pytest tests/test_engine.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.transcriber import WhisperTranscriber

pytestmark = pytest.mark.skipif(
    os.environ.get("WHISPER_RUN_ENGINE") != "1",
    reason="frozen clip, real engine",
)


def test_frozen_short_clip_matches_this_engine() -> None:
    engine = WhisperTranscriber("small", device="cpu", compute_type="int8", cpu_threads=4)
    engine.load()
    assert engine.ready()
    transcript = engine.transcribe(Path("tests/fixtures/short.wav"), None)
    expected = json.loads(Path("tests/fixtures/short.expected.json").read_text(encoding="utf-8"))
    assert {"text": transcript.text, "language": transcript.language} == expected
