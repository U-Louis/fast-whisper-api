"""Process configuration. Secrets are hashes, and the process refuses a bad set."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

MAX_BYTES = 5 * 1024 * 1024
MULTIPART_SLACK = 64 * 1024
PROBE_TIMEOUT_SECONDS = 10.0
MAX_TOKEN_LENGTH = 512
KEY_ID = re.compile(r"[a-z][a-z0-9-]{0,31}")
CONFIG_NAME = "whisper-api.conf"
PROBE_DURATION_KEY = "probe_duration_seconds"
_DEVICES = {"cpu": "int8", "cuda": "float16"}
_DEFAULT_CPU_THREADS = 4


@dataclass(frozen=True)
class ApiKey:
    id: str
    sha256_hex: str

    def digest(self) -> bytes:
        return bytes.fromhex(self.sha256_hex)


@dataclass(frozen=True)
class Settings:
    keys: tuple[ApiKey, ...]
    model: str
    max_bytes: int
    max_seconds: float
    multipart_slack: int
    tmp_dir: Path
    probe_timeout_seconds: float
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = _DEFAULT_CPU_THREADS

    @classmethod
    def from_env(cls) -> Settings:
        raw = os.environ.get("WHISPER_KEYS", "")
        model = os.environ.get("WHISPER_MODEL", "small")
        tmp = Path(os.environ.get("WHISPER_TMP_DIR", "/tmp/whisper-api"))
        keys = _parse_keys(raw)
        if model != "small":
            _refuse("WHISPER_MODEL must be small")
        device, compute_type, cpu_threads = _engine_from_env()
        return cls(
            keys=keys,
            model=model,
            max_bytes=MAX_BYTES,
            max_seconds=read_probe_duration(config_path()),
            multipart_slack=MULTIPART_SLACK,
            tmp_dir=tmp,
            probe_timeout_seconds=PROBE_TIMEOUT_SECONDS,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )


def config_path() -> Path:
    raw = os.environ.get("WHISPER_CONFIG", "").strip()
    if raw:
        return Path(raw)
    return Path(CONFIG_NAME)


def read_probe_duration(path: Path) -> float:
    """The longest clip the duration probe accepts, in seconds. Only this file sets it."""
    if not path.is_file():
        _refuse(f"config file not found: {path}")
    found: float | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        _refuse("config file is not utf-8")
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        key, sep, value = text.partition("=")
        if not sep or key.strip() != PROBE_DURATION_KEY:
            _refuse(f"config file must set {PROBE_DURATION_KEY}")
        if found is not None:
            _refuse(f"{PROBE_DURATION_KEY} is set more than once")
        found = _positive_seconds(value.strip())
    if found is None:
        _refuse(f"{PROBE_DURATION_KEY} is required")
    return found


def _engine_from_env() -> tuple[str, str, int]:
    device = _device(os.environ.get("WHISPER_DEVICE"))
    compute_type = _compute_type(device, os.environ.get("WHISPER_COMPUTE_TYPE"))
    cpu_threads = _cpu_threads(os.environ.get("WHISPER_CPU_THREADS"))
    return device, compute_type, cpu_threads


def _device(raw: str | None) -> str:
    if raw is None:
        return "cpu"
    value = raw.strip().lower()
    if value not in _DEVICES:
        _refuse("WHISPER_DEVICE must be cpu or cuda")
    return value


def _compute_type(device: str, raw: str | None) -> str:
    expected = _DEVICES[device]
    if raw is None:
        return expected
    value = raw.strip().lower()
    if value != expected:
        _refuse(f"WHISPER_COMPUTE_TYPE must be {expected} when WHISPER_DEVICE is {device}")
    return value


def _cpu_threads(raw: str | None) -> int:
    if raw is None:
        return _DEFAULT_CPU_THREADS
    text = raw.strip()
    if not text.isdigit():
        _refuse("WHISPER_CPU_THREADS must be a positive integer")
    value = int(text)
    if value < 1:
        _refuse("WHISPER_CPU_THREADS must be a positive integer")
    return value


def _positive_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        _refuse(f"{PROBE_DURATION_KEY} must be a positive number of seconds")
    if value <= 0 or value != value or value in (float("inf"), float("-inf")):
        _refuse(f"{PROBE_DURATION_KEY} must be a positive number of seconds")
    return value


def _parse_keys(raw: str) -> tuple[ApiKey, ...]:
    if not raw.strip():
        _refuse("WHISPER_KEYS is required")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        _refuse("WHISPER_KEYS is required")
    parsed: list[ApiKey] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for part in parts:
        key_id, sep, digest = part.partition(":")
        if not sep or not KEY_ID.fullmatch(key_id):
            _refuse("WHISPER_KEYS has an invalid key id")
        hexdigest = digest.strip().lower()
        if len(hexdigest) != 64 or any(c not in "0123456789abcdef" for c in hexdigest):
            _refuse("WHISPER_KEYS has an invalid hash")
        if key_id in seen_ids or hexdigest in seen_hashes:
            _refuse("WHISPER_KEYS must be distinct")
        seen_ids.add(key_id)
        seen_hashes.add(hexdigest)
        parsed.append(ApiKey(key_id, hexdigest))
    return tuple(parsed)


def _refuse(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)
