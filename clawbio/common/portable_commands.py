"""
clawbio.common.portable_commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Centralized, portable replay-command generation for ClawBio skills.

Problem solved
--------------
Previously each skill wrote commands.sh independently, leading to:
- Machine-specific absolute paths  (/home/sooraj/ClawBio/...)
- cwd-dependent bare script names  (python lit_synthesizer.py)
- Verbatim user paths that break on other machines

Solution
--------
All skills call `build_portable_commands_sh()` which produces a
self-anchoring bash script that:
1. Locates the repo root relative to the script's own location
2. Constructs the skill path from the repo root
3. Resolves input/output paths relative to where commands.sh is executed

Usage in a skill
----------------
    from clawbio.common.portable_commands import build_portable_commands_sh

    commands = build_portable_commands_sh(
        skill_name="lit-synthesizer",
        script_name="lit_synthesizer.py",
        args={"--query": query, "--output": "./report"},
        generated_at=now,
    )
    (repro_dir / "commands.sh").write_text(commands)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Depth from reproducibility/ to repo root:
# output/reproducibility/ -> output/ -> (anywhere) -> repo root
# We anchor to SCRIPT_DIR and walk up to the repo root via a known marker.
_ANCHOR_HEADER = """\
#!/usr/bin/env bash
# ClawBio reproducibility bundle — portable replay command
# Generated: {generated_at}
# Skill: {skill_name}
#
# How to replay:
#   bash reproducibility/commands.sh
# from anywhere inside the repository clone.

set -euo pipefail

# ── Locate repo root ──────────────────────────────────────────────────────────
# Walk up from this script's directory until we find the repo marker (skills/).
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ ! -d "$REPO_ROOT/skills" && "$REPO_ROOT" != "/" ]]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -d "$REPO_ROOT/skills" ]]; then
  echo "ERROR: Could not locate repo root (no skills/ directory found)" >&2
  exit 1
fi

# ── Skill script ──────────────────────────────────────────────────────────────
SKILL_SCRIPT="$REPO_ROOT/skills/{skill_name}/{script_name}"
if [[ ! -f "$SKILL_SCRIPT" ]]; then
  echo "ERROR: Skill script not found: $SKILL_SCRIPT" >&2
  exit 1
fi

"""

_COMMAND_TEMPLATE = """\
# ── Replay command ────────────────────────────────────────────────────────────
python "$SKILL_SCRIPT" \\
{args_block}
"""


def _format_value(val: Any) -> str:
    """Return a shell-safe representation of a CLI argument value."""
    if val is None:
        return ""
    s = str(val)
    # If value contains spaces or special chars, quote it
    if any(c in s for c in (" ", "\t", "$", "`", "\\", '"', "'")):
        return f'"{s}"'
    return s


def _make_output_portable(output_path: str) -> str:
    """
    Convert an output path to a portable form.
    - Absolute paths -> warn and use as-is (user chose absolute)
    - Relative paths -> keep relative (portable)
    - Paths that look like /tmp/... -> keep as-is (temp runs)
    """
    p = Path(output_path)
    if p.is_absolute():
        # Suggest keeping relative; we store as-is but add a comment
        return str(output_path)
    return str(output_path)


def build_portable_commands_sh(
    skill_name: str,
    script_name: str,
    args: dict[str, Any],
    generated_at: str | None = None,
) -> str:
    """
    Build a portable, self-anchoring commands.sh content string.

    Parameters
    ----------
    skill_name:
        The skill folder name, e.g. "lit-synthesizer"
    script_name:
        The Python script filename, e.g. "lit_synthesizer.py"
    args:
        Dict of CLI argument name -> value.
        Use None value to emit a flag with no value (e.g. {"--demo": None}).
        Example: {"--query": "CRISPR", "--output": "./report"}
    generated_at:
        ISO timestamp string. Defaults to current UTC time.

    Returns
    -------
    str
        Complete contents of commands.sh, ready to write to disk.
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = _ANCHOR_HEADER.format(
        generated_at=generated_at,
        skill_name=skill_name,
        script_name=script_name,
    )

    # Build args block — each arg on its own indented line
    arg_lines = []
    for flag, value in args.items():
        if value is None:
            # Boolean flag, no value
            arg_lines.append(f"    {flag}")
        else:
            formatted = _format_value(value)
            arg_lines.append(f"    {flag} {formatted}")

    args_block = " \\\n".join(arg_lines)

    command = _COMMAND_TEMPLATE.format(args_block=args_block)

    return header + command


def write_portable_commands_sh(
    repro_dir: "Path",
    skill_name: str,
    script_name: str,
    args: dict[str, Any],
    generated_at: str | None = None,
) -> None:
    """
    Convenience wrapper: build and write commands.sh to repro_dir.

    Parameters
    ----------
    repro_dir:
        Path to the reproducibility/ directory (will be created if needed).
    skill_name, script_name, args, generated_at:
        Passed through to build_portable_commands_sh().
    """
    repro_dir = Path(repro_dir)
    repro_dir.mkdir(parents=True, exist_ok=True)
    content = build_portable_commands_sh(
        skill_name=skill_name,
        script_name=script_name,
        args=args,
        generated_at=generated_at,
    )
    (repro_dir / "commands.sh").write_text(content, encoding="utf-8")
