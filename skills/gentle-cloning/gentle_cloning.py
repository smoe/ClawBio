#!/usr/bin/env python3
"""ClawBio wrapper for structured GENtle CLI requests."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

REQUEST_SCHEMA = "gentle.clawbio_skill_request.v1"
RESULT_SCHEMA = "gentle.clawbio_skill_result.v1"


class SkillError(RuntimeError):
    """Deterministic wrapper error shown in report.md/result.json."""


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_request(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SkillError(f"invalid JSON in '{path}': {e}") from e
    if not isinstance(payload, dict):
        raise SkillError("request JSON must be an object")
    schema = payload.get("schema")
    if schema != REQUEST_SCHEMA:
        raise SkillError(f"unsupported request schema '{schema}', expected '{REQUEST_SCHEMA}'")
    return payload


def _demo_request() -> dict[str, Any]:
    return {
        "schema": REQUEST_SCHEMA,
        "mode": "raw",
        "raw_args": ["capabilities"],
        "timeout_secs": 180,
    }


def _resolve_cli(explicit: str | None) -> list[str]:
    if explicit:
        return shlex.split(explicit)
    env_cmd = os.environ.get("GENTLE_CLI_CMD", "").strip()
    if env_cmd:
        return shlex.split(env_cmd)
    path_hit = shutil.which("gentle_cli")
    if path_hit:
        return [path_hit]
    raise SkillError(
        "Could not resolve gentle_cli. Set GENTLE_CLI_CMD, pass --gentle-cli, "
        "or install gentle_cli on PATH."
    )


def _request_to_args(request: dict[str, Any]) -> list[str]:
    mode = str(request.get("mode", "")).strip()
    if not mode:
        raise SkillError("request missing required field 'mode'")
    if mode == "raw":
        raw_args = request.get("raw_args")
        if not isinstance(raw_args, list) or not raw_args:
            raise SkillError("mode=raw requires non-empty string array 'raw_args'")
        if not all(isinstance(v, str) and v for v in raw_args):
            raise SkillError("mode=raw 'raw_args' must contain non-empty strings")
        return raw_args
    if mode == "shell":
        shell_line = request.get("shell_line")
        if not isinstance(shell_line, str) or not shell_line.strip():
            raise SkillError("mode=shell requires non-empty string field 'shell_line'")
        return ["shell", shell_line.strip()]
    raise SkillError("unsupported mode. Use raw or shell")


def _parse_structured_stdout(stdout: str) -> dict[str, Any]:
    """Return structured GENtle result fields emitted as JSON on stdout."""
    text = stdout.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_report(
    path: Path,
    *,
    request: dict[str, Any],
    command: list[str] | None,
    status: str,
    started_utc: str,
    ended_utc: str,
    stdout: str,
    stderr: str,
    error: str | None,
) -> None:
    command_text = " ".join(shlex.quote(v) for v in command) if command else "(none)"
    lines = [
        "# GENtle ClawBio Skill Report",
        "",
        f"- Started (UTC): `{started_utc}`",
        f"- Ended (UTC): `{ended_utc}`",
        f"- Status: `{status}`",
        f"- Mode: `{request.get('mode', '')}`",
    ]
    if error:
        lines.append(f"- Error: `{error}`")
    lines.extend(
        [
            "",
            "## Command",
            "",
            "```bash",
            command_text,
            "```",
            "",
            "## Stdout",
            "",
            "```text",
            stdout.rstrip(),
            "```",
            "",
            "## Stderr",
            "",
            "```text",
            stderr.rstrip(),
            "```",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute a structured GENtle request and write a ClawBio bundle."
    )
    parser.add_argument("--input", help="Path to gentle.clawbio_skill_request.v1 JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run a capabilities demo")
    parser.add_argument("--gentle-cli", help="Explicit command used to invoke gentle_cli")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = _now_utc_iso()
    ended = started
    status = "failed"
    error: str | None = None
    request = _demo_request()
    command: list[str] | None = None
    stdout = ""
    stderr = ""
    exit_code: int | None = None

    try:
        if args.demo:
            request = _demo_request()
        elif args.input:
            request = _read_request(Path(args.input))
        else:
            raise SkillError("--input is required unless --demo is used")

        timeout_secs = int(request.get("timeout_secs", 180))
        if timeout_secs <= 0:
            raise SkillError("timeout_secs must be > 0")
        command = _resolve_cli(args.gentle_cli) + _request_to_args(request)
        run = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            check=False,
        )
        stdout = run.stdout
        stderr = run.stderr
        exit_code = run.returncode
        status = "ok" if run.returncode == 0 else "command_failed"
        if run.returncode != 0:
            error = f"gentle_cli exited with {run.returncode}; inspect stderr in report.md"
    except subprocess.TimeoutExpired as e:
        error = f"command timed out after {e.timeout} seconds"
        status = "timeout"
    except SkillError as e:
        error = str(e)
        status = "failed"
    except Exception as e:  # pragma: no cover - defensive boundary
        error = f"unexpected error: {type(e).__name__}: {e}"
        status = "failed"
    finally:
        ended = _now_utc_iso()

    report_path = output_dir / "report.md"
    result_path = output_dir / "result.json"
    _write_report(
        report_path,
        request=request,
        command=command,
        status=status,
        started_utc=started,
        ended_utc=ended,
        stdout=stdout,
        stderr=stderr,
        error=error,
    )

    structured_stdout = _parse_structured_stdout(stdout)
    result_payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "request": request,
        "started_utc": started,
        "ended_utc": ended,
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "artifacts": {
            "report_md": str(report_path),
            "result_json": str(result_path),
        },
    }
    # GENtle owns chat summaries, preferred artifacts, and suggested actions.
    # If gentle_cli emits those structured fields, preserve them verbatim in the
    # wrapper result so ClawBio can offer them without reading prose.
    for key in ("chat_summary_lines", "preferred_artifacts", "suggested_actions", "report_md"):
        if key in structured_stdout:
            result_payload[key] = structured_stdout[key]

    result_path.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result_payload, indent=2, ensure_ascii=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
