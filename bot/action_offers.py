"""Shared helpers for rendering and executing structured skill follow-ups."""

from __future__ import annotations

import difflib
import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

DEFAULT_PENDING_ACTION_TTL_SECONDS = 30 * 60

# This module sits between a structured skill result and the chat adapters.
# Its job is intentionally narrow:
# 1. validate/sanitize `suggested_actions[]` emitted by a skill
# 2. render those actions into a compact conversational offer
# 3. interpret a later human reply against the stored actions
# 4. execute only the stored structured request for the matched action
#
# That last point is the safety boundary: we do not reconstruct commands from
# prose and we do not let the user's reply inject new shell input.
#
# Structured field semantics expected from a skill result:
# - chat_summary_lines: concise, skill-authored status text for chat replies
# - preferred_artifacts: files the chat/web UI should prioritize for display
# - suggested_actions: deterministic follow-up choices with nested requests
# - report_md: full markdown report text, embedded for convenient rendering

# Replies like "yes" or "go ahead" only make sense when a single action is pending.
# We keep these separate from indexed confirmations so a plain "yes" can stay
# deliberately ambiguous when multiple actions are on offer.
_YES_ONLY = {
    "yes",
    "y",
    "yeah",
    "yep",
    "ok",
    "okay",
    "sure",
    "please do",
    "go ahead",
    "do it",
    "run it",
}

_CANCEL_WORDS = {
    "cancel",
    "no",
    "nope",
    "never mind",
    "nevermind",
    "stop",
    "abort",
}

# Prefix verbs that may appear before a numeric selection, e.g. "run 2".
# This is intentionally a different set from _YES_ONLY: it models explicit
# indexed confirmations rather than stand-alone approvals.
_INDEX_CONFIRM_PREFIXES = (
    "yes",
    "run",
    "do",
    "execute",
    "please run",
    "please do",
)
_INDEX_CONFIRM_PATTERN = re.compile(
    r"\s*(?:" + "|".join(re.escape(prefix) for prefix in _INDEX_CONFIRM_PREFIXES) + r")?\s*(\d+)\s*",
    flags=re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    # Matching should be stable across "Yes", " yes ", and multi-space input.
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_action_offer(result: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return chat-presentable suggested actions from a skill result."""
    actions = result.get("suggested_actions")
    if not isinstance(actions, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        label = action.get("label")
        action_id = action.get("action_id")
        request = action.get("request")
        # We keep the validation deliberately lightweight here:
        # enough to trust the entry for chat presentation and later execution,
        # without forcing every optional field to be present. The nested
        # request is required because confirmation executes that object; we do
        # not fall back to shell_line or prose.
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(action_id, str) or not action_id.strip():
            continue
        if not isinstance(request, dict):
            continue
        cleaned.append(action)
    return cleaned or None


def extract_chat_summary_lines(result: dict[str, Any]) -> list[str]:
    """Return concise summary lines intended for chat before any full report."""
    lines = result.get("chat_summary_lines")
    if not isinstance(lines, list):
        return []
    return [str(line).strip() for line in lines if str(line).strip()]


def load_bundle_fields(output_dir: str | Path | None) -> dict[str, Any]:
    """Read the structured fields a chat adapter needs from a skill bundle."""
    if output_dir is None:
        return {}

    out_dir = Path(output_dir)
    if not out_dir.exists():
        return {}

    fields: dict[str, Any] = {}
    result_json_path = out_dir / "result.json"
    if result_json_path.exists():
        fields["result_json_path"] = str(result_json_path)
        try:
            payload = json.loads(result_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A malformed result.json should not break chat delivery; the
            # markdown report fallback below can still give the user something.
            payload = None
        if isinstance(payload, dict):
            fields["skill_result_json"] = payload
            # Mirror the runner's promoted field contract for bot subprocesses.
            # The adapters execute skills through the CLI, then recover these
            # fields from the bundle afterward for chat rendering.
            for key in (
                "chat_summary_lines",
                "preferred_artifacts",
                "suggested_actions",
                "report_md",
            ):
                if key in payload:
                    fields[key] = payload[key]

    if "report_md" not in fields:
        # Older skills may only produce markdown. Treat that as display text,
        # but prefer structured report_md when a skill provided it explicitly.
        for pattern in ("report.md", "*_report.md", "*.md"):
            for md_file in sorted(out_dir.glob(pattern)):
                if md_file.name.startswith("."):
                    continue
                try:
                    fields["report_md"] = md_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                return fields
    return fields


def render_action_offer(actions: list[dict[str, Any]]) -> str:
    """Render numbered follow-up choices for Telegram/Discord text replies."""
    if not actions:
        return ""
    lines = ["I can do the next step for you:"]
    for idx, action in enumerate(actions, start=1):
        suffix = ""
        if action.get("requires_confirmation") is False:
            suffix = " (safe refresh)"
        lines.append(f"{idx}. {action.get('label', f'Action {idx}')}{suffix}")
    if len(actions) == 1:
        lines.append("Reply with `yes`, `1`, the action name, or `cancel`.")
    else:
        choices = ", ".join(f"`{i}`" for i in range(1, len(actions) + 1))
        lines.append(f"Reply with {choices}, `yes 1`, the action name, or `cancel`.")
    return "\n".join(lines)


def is_cancel_reply(reply_text: str) -> bool:
    """Return True when a reply clearly declines the pending offer."""
    return _normalize_text(reply_text) in _CANCEL_WORDS


def _match_by_index(reply_text: str, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match replies like '1', 'run 2', or 'yes 3' to a stored action."""
    m = _INDEX_CONFIRM_PATTERN.fullmatch(reply_text.strip())
    if not m:
        return None
    idx = int(m.group(1)) - 1
    if 0 <= idx < len(actions):
        return actions[idx]
    return None


def _match_by_label(reply_text: str, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match a reply against action labels or ids without inventing new work."""
    normalized = _normalize_text(reply_text)
    exact: list[dict[str, Any]] = []
    loose: list[dict[str, Any]] = []
    by_norm: dict[str, dict[str, Any]] = {}
    for action in actions:
        label = _normalize_text(str(action.get("label", "")))
        action_id = _normalize_text(str(action.get("action_id", "")))
        by_norm[label] = action
        by_norm[action_id] = action
        if normalized in {label, action_id}:
            exact.append(action)
        if normalized and (normalized in label or label in normalized):
            loose.append(action)
    if len(exact) == 1:
        return exact[0]
    if len(loose) == 1:
        return loose[0]

    # For human-friendly chat replies we allow one final fuzzy match, but only
    # when there is a single strong candidate. This keeps "prepare human" or a
    # near-exact label typo usable without guessing between multiple actions.
    labels = [str(action.get("label", "")) for action in actions]
    label_map = {_normalize_text(label): action for label, action in zip(labels, actions)}
    close = difflib.get_close_matches(normalized, list(label_map.keys()), n=2, cutoff=0.86)
    if len(close) == 1:
        return label_map[close[0]]
    return None


def parse_action_reply(reply_text: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify a follow-up reply against stored actions."""
    if not actions:
        return {"kind": "none"}
    normalized = _normalize_text(reply_text)
    if not normalized:
        return {"kind": "none"}
    if is_cancel_reply(normalized):
        return {"kind": "cancel"}

    # Prefer explicit numeric choices first because they are unambiguous and
    # work well across adapters ("1", "run 2", "yes 3").
    index_match = _match_by_index(reply_text, actions)
    if index_match is not None:
        return {"kind": "matched", "action": index_match}

    # A plain "yes" is only enough when there was exactly one offered action.
    # With multiple actions we intentionally return "ambiguous" so the caller
    # can ask a short follow-up instead of guessing.
    if normalized in _YES_ONLY:
        if len(actions) == 1:
            return {"kind": "matched", "action": actions[0]}
        return {"kind": "ambiguous"}

    label_match = _match_by_label(reply_text, actions)
    if label_match is not None:
        return {"kind": "matched", "action": label_match}

    return {"kind": "none"}


def make_pending_action_entry(
    *,
    skill: str,
    actions: list[dict[str, Any]],
    source_summary: list[str] | None = None,
    source_output_dir: str | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    # Store the original offered actions verbatim so later confirmation can run
    # one of those exact structured requests and nothing else.
    return {
        "skill": skill,
        "timestamp": float(timestamp if timestamp is not None else time.time()),
        "actions": actions,
        "source_summary": list(source_summary or []),
        "source_output_dir": source_output_dir,
    }


def is_pending_action_expired(
    entry: dict[str, Any],
    *,
    now_ts: float | None = None,
    ttl_seconds: int = DEFAULT_PENDING_ACTION_TTL_SECONDS,
) -> bool:
    """Return True when a stored offer is too old to accept safely."""
    try:
        ts = float(entry.get("timestamp", 0))
    except (TypeError, ValueError):
        return True
    now = float(now_ts if now_ts is not None else time.time())
    return now - ts > ttl_seconds


def choice_list_text(actions: list[dict[str, Any]]) -> str:
    """Return a human-readable list of valid numeric choices."""
    choices = [str(i) for i in range(1, len(actions) + 1)]
    if not choices:
        return ""
    if len(choices) == 1:
        return choices[0]
    if len(choices) == 2:
        return f"{choices[0]} or {choices[1]}"
    return ", ".join(choices[:-1]) + f", or {choices[-1]}"


def execute_stored_action(
    pending_entry: dict[str, Any],
    action: dict[str, Any],
    *,
    runner: Callable[..., dict[str, Any]],
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize a stored request and execute it via the provided runner."""
    request = action.get("request")
    if not isinstance(request, dict):
        raise ValueError("stored action is missing a structured request payload")
    skill = str(pending_entry.get("skill") or "").strip()
    if not skill:
        raise ValueError("pending action entry is missing the source skill")

    with tempfile.TemporaryDirectory(prefix="clawbio_action_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        request_path = tmpdir_path / "request.json"
        # The runner already knows how to execute a ClawBio skill from an input
        # JSON file, so the safest path is to materialize the stored nested
        # request exactly as-is and hand it back to the normal runner flow.
        request_path.write_text(
            json.dumps(request, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_root) / f"{skill}_{ts}"
        runner_kwargs: dict[str, Any] = {
            "skill_name": skill,
            "input_path": str(request_path),
            "output_dir": str(output_dir),
            "demo": False,
        }
        timeout_secs = action.get("timeout_secs")
        # Per-action timeouts come from the structured skill result, so the
        # follow-up can inherit a longer budget for explicit prepare/sync steps.
        if isinstance(timeout_secs, int) and timeout_secs > 0:
            runner_kwargs["timeout"] = timeout_secs
        result = runner(
            **runner_kwargs,
        )

    return result
