import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.settings import MAX_BYTES, Settings, read_probe_duration


def test_defaults() -> None:
    assert MAX_BYTES == 5 * 1024 * 1024
    assert read_probe_duration(Path("whisper-api.conf")) == 30


def test_any_number_of_distinct_keys(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    conf = tmp_path / "whisper-api.conf"
    conf.write_text("probe_duration_seconds = 12.5\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_CONFIG", str(conf))
    one = "ab" * 32
    monkeypatch.setenv("WHISPER_KEYS", f"app-a:{one}")
    assert Settings.from_env().max_seconds == 12.5
    assert [key.id for key in Settings.from_env().keys] == ["app-a"]
    many = ",".join(f"app-{name}:{'%02x' % index * 32}" for index, name in enumerate("abc"))
    monkeypatch.setenv("WHISPER_KEYS", many)
    assert [key.id for key in Settings.from_env().keys] == ["app-a", "app-b", "app-c"]


def test_rejects_empty_and_duplicate_keys(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    conf = tmp_path / "whisper-api.conf"
    conf.write_text("probe_duration_seconds = 30\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_CONFIG", str(conf))
    monkeypatch.setenv("WHISPER_KEYS", "")
    with pytest.raises(SystemExit):
        Settings.from_env()
    digest = "ab" * 32
    monkeypatch.setenv("WHISPER_KEYS", f"app-a:{digest},app-b:{digest}")
    with pytest.raises(SystemExit):
        Settings.from_env()


def test_probe_duration_must_be_a_positive_number(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    conf = tmp_path / "whisper-api.conf"
    monkeypatch.setenv("WHISPER_CONFIG", str(conf))
    monkeypatch.setenv("WHISPER_KEYS", f"app-a:{'ab' * 32}")
    conf.write_text("probe_duration_seconds = 0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        Settings.from_env()
    monkeypatch.setenv("WHISPER_CONFIG", str(tmp_path / "missing.conf"))
    with pytest.raises(SystemExit):
        Settings.from_env()


def test_rejects_other_models(monkeypatch: pytest.MonkeyPatch) -> None:
    digest_a = "ab" * 32
    digest_b = "cd" * 32
    monkeypatch.setenv("WHISPER_KEYS", f"app-a:{digest_a},app-b:{digest_b}")
    monkeypatch.setenv("WHISPER_MODEL", "turbo")
    with pytest.raises(SystemExit):
        Settings.from_env()


def test_engine_defaults_to_cpu_int8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conf = tmp_path / "whisper-api.conf"
    conf.write_text("probe_duration_seconds = 30\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_CONFIG", str(conf))
    monkeypatch.setenv("WHISPER_KEYS", f"app-a:{'ab' * 32}")
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("WHISPER_CPU_THREADS", raising=False)
    settings = Settings.from_env()
    assert settings.device == "cpu"
    assert settings.compute_type == "int8"
    assert settings.cpu_threads == 4


def test_cuda_uses_float16(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conf = tmp_path / "whisper-api.conf"
    conf.write_text("probe_duration_seconds = 30\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_CONFIG", str(conf))
    monkeypatch.setenv("WHISPER_KEYS", f"app-a:{'ab' * 32}")
    monkeypatch.setenv("WHISPER_DEVICE", "CUDA")
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.setenv("WHISPER_CPU_THREADS", "8")
    settings = Settings.from_env()
    assert settings.device == "cuda"
    assert settings.compute_type == "float16"
    assert settings.cpu_threads == 8
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float16")
    assert Settings.from_env().compute_type == "float16"


def test_rejects_a_mismatched_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conf = tmp_path / "whisper-api.conf"
    conf.write_text("probe_duration_seconds = 30\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_CONFIG", str(conf))
    monkeypatch.setenv("WHISPER_KEYS", f"app-a:{'ab' * 32}")
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float16")
    with pytest.raises(SystemExit):
        Settings.from_env()
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    with pytest.raises(SystemExit):
        Settings.from_env()
    monkeypatch.setenv("WHISPER_DEVICE", "gpu")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float16")
    with pytest.raises(SystemExit):
        Settings.from_env()
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("WHISPER_CPU_THREADS", "0")
    with pytest.raises(SystemExit):
        Settings.from_env()


def test_process_refuses_to_import_without_keys() -> None:
    env = os.environ.copy()
    env.pop("WHISPER_KEYS", None)
    result = subprocess.run(
        [sys.executable, "-c", "import app.asgi"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "WHISPER_KEYS is required" in result.stderr
    assert "Bearer" not in result.stderr


def test_process_refuses_cpu_with_float16() -> None:
    env = os.environ.copy()
    env["WHISPER_KEYS"] = "app-a:" + ("ab" * 32)
    env["WHISPER_DEVICE"] = "cpu"
    env["WHISPER_COMPUTE_TYPE"] = "float16"
    result = subprocess.run(
        [sys.executable, "-c", "import app.asgi"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "WHISPER_COMPUTE_TYPE must be int8 when WHISPER_DEVICE is cpu" in result.stderr
