"""faster-whisper behind a small seam so tests do not load the model."""

from __future__ import annotations

import logging
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.errors import InvalidLanguage

log = logging.getLogger("whisper_api")

_DECODE = {
    "beam_size": 1,
    "task": "transcribe",
    "temperature": 0,
}
_PAIRS = {"cpu": "int8", "cuda": "float16"}


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str


class Transcriber(Protocol):
    def ready(self) -> bool: ...

    def transcribe(self, path: Path, language: str | None) -> Transcript: ...


class WhisperTranscriber:
    def __init__(self, model_name: str, device: str, compute_type: str, cpu_threads: int) -> None:
        if model_name != "small":
            raise RuntimeError("model must be small")
        expected = _PAIRS.get(device)
        if expected is None or compute_type != expected:
            raise RuntimeError("cpu uses int8 and cuda uses float16")
        if cpu_threads < 1:
            raise RuntimeError("cpu_threads must be a positive integer")
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._model: object | None = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
            )
            _warmup(model)
        except Exception:
            log.exception("model_load_failed", extra={"event": "model_load", "outcome": "failed"})
            self._model = None
            return
        self._model = model

    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, path: Path, language: str | None) -> Transcript:
        model = self._model
        if model is None:
            raise RuntimeError("model is not loaded")
        try:
            text, detected = _consume(model, path, language)
        except (ValueError, KeyError):
            raise InvalidLanguage() from None
        chosen = language if language is not None else detected
        if not isinstance(chosen, str) or not chosen:
            raise InvalidLanguage()
        return Transcript(text=text, language=chosen)


def _consume(model: object, path: Path, language: str | None) -> tuple[str, str | None]:
    kwargs: dict[str, object] = dict(_DECODE)
    if language is not None:
        kwargs["language"] = language
    segments, info = model.transcribe(str(path), **kwargs)  # type: ignore[attr-defined]
    text = "".join(str(getattr(segment, "text", "")) for segment in segments).strip()
    detected = getattr(info, "language", None)
    return text, detected if isinstance(detected, str) else None


def _warmup(model: object) -> None:
    """One silent clip, so the first client does not pay model warmup. The text is discarded."""
    fd, name = tempfile.mkstemp(prefix="whisper-warmup-", suffix=".wav")
    os.close(fd)
    path = Path(name)
    try:
        _write_silence(path)
        _consume(model, path, None)
    finally:
        path.unlink(missing_ok=True)


def _write_silence(path: Path) -> None:
    rate = 16000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * rate)
