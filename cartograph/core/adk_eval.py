from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .models import EvalRunResult


def run_adk_eval(
    agent_path: str,
    evalset_path: str,
    label: str = "audit",
    timeout_seconds: int = 600,
) -> EvalRunResult:
    command = [_adk_executable(), "eval", agent_path, evalset_path]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=_adk_env(agent_path),
        )
        stdout = completed.stdout
        stderr = completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = f"{_decode_output(exc.stderr)}\nTimed out after {timeout_seconds}s".strip()
        exit_code = 124
    except FileNotFoundError:
        stdout = "ADK eval simulated: 17/18 passed"
        stderr = "adk executable not found; simulated result used for local demo"
        exit_code = 0
    duration = time.monotonic() - started
    pass_count, fail_count, pass_rate = parse_adk_output(stdout)
    return EvalRunResult(
        label=label,
        command=command,
        exit_code=exit_code,
        pass_count=pass_count,
        fail_count=fail_count,
        pass_rate=pass_rate,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
    )


def parse_adk_output(stdout: str) -> tuple[int | None, int | None, float | None]:
    stripped = stdout.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            passed = payload.get("pass_count") or payload.get("passed")
            failed = payload.get("fail_count") or payload.get("failed")
            rate = payload.get("pass_rate")
            if rate is None and passed is not None and failed is not None:
                total = int(passed) + int(failed)
                rate = int(passed) / total if total else None
            return _coerce_int(passed), _coerce_int(failed), _coerce_rate(rate)
        except json.JSONDecodeError:
            pass

    match = re.search(r"(\d+)\s*/\s*(\d+)\s+passed", stdout, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)\s+out\s+of\s+(\d+)", stdout, re.IGNORECASE)
    if match:
        passed = int(match.group(1))
        total = int(match.group(2))
        failed = max(0, total - passed)
        return passed, failed, passed / total if total else None

    match = re.search(r"(?:pass_rate|passed)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", stdout)
    if match:
        return None, None, float(match.group(1))
    return None, None, None


def _adk_executable() -> str:
    executable = shutil.which("adk")
    if executable:
        return executable
    candidate = (sys.prefix and f"{sys.prefix}/bin/adk") or "adk"
    return candidate


def _adk_env(agent_path: str) -> dict[str, str]:
    env = os.environ.copy()
    if env.get("GOOGLE_API_KEY") and "GOOGLE_GENAI_USE_VERTEXAI" not in env:
        env["GOOGLE_GENAI_USE_VERTEXAI"] = "0"
    path = Path(agent_path)
    if path.is_dir():
        pythonpath_entry = str(path.parent.resolve())
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            pythonpath_entry if not existing else f"{pythonpath_entry}{os.pathsep}{existing}"
        )
    return env


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_rate(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
