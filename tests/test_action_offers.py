from __future__ import annotations

import json
from pathlib import Path

from bot.action_offers import (
    execute_stored_action,
    extract_action_offer,
    is_pending_action_expired,
    load_bundle_fields,
    make_pending_action_entry,
    parse_action_reply,
    render_action_offer,
)


def _demo_actions() -> list[dict]:
    return [
        {
            "action_id": "prepare-grch38",
            "label": "Prepare Human GRCh38 Ensembl 116",
            "kind": "request",
            "request": {"schema": "gentle.request.v1", "command": "genomes prepare grch38"},
            "requires_confirmation": True,
            "timeout_secs": 123,
        },
        {
            "action_id": "sync-attract",
            "label": "Sync ATtRACT runtime snapshot",
            "kind": "request",
            "request": {"schema": "gentle.request.v1", "command": "resources sync attract"},
            "requires_confirmation": False,
        },
    ]


def test_render_action_offer_lists_choices_and_safe_refresh_hint():
    rendered = render_action_offer(_demo_actions())
    assert "I can do the next step for you:" in rendered
    assert "1. Prepare Human GRCh38 Ensembl 116" in rendered
    assert "2. Sync ATtRACT runtime snapshot (safe refresh)" in rendered
    assert "Reply with `1`, `2`" in rendered
    assert "cancel" in rendered


def test_extract_action_offer_requires_structured_request():
    actions = _demo_actions() + [
        {
            "action_id": "shell-only",
            "label": "Do not offer shell-only action",
            "shell_line": "gentle resources sync attract",
        }
    ]

    extracted = extract_action_offer({"suggested_actions": actions})

    assert extracted == _demo_actions()


def test_parse_action_reply_matches_numeric_choice():
    parsed = parse_action_reply("1", _demo_actions())
    assert parsed["kind"] == "matched"
    assert parsed["action"]["action_id"] == "prepare-grch38"


def test_parse_action_reply_matches_yes_with_index():
    parsed = parse_action_reply("yes 1", _demo_actions())
    assert parsed["kind"] == "matched"
    assert parsed["action"]["label"] == "Prepare Human GRCh38 Ensembl 116"


def test_parse_action_reply_matches_exact_label():
    parsed = parse_action_reply("Sync ATtRACT runtime snapshot", _demo_actions())
    assert parsed["kind"] == "matched"
    assert parsed["action"]["action_id"] == "sync-attract"


def test_parse_action_reply_marks_plain_yes_ambiguous_for_multiple_actions():
    parsed = parse_action_reply("yes", _demo_actions())
    assert parsed["kind"] == "ambiguous"


def test_parse_action_reply_handles_cancel():
    parsed = parse_action_reply("never mind", _demo_actions())
    assert parsed["kind"] == "cancel"


def test_pending_action_entry_expires_after_ttl():
    entry = make_pending_action_entry(
        skill="gentle-cloning",
        actions=_demo_actions(),
        timestamp=100.0,
    )
    assert is_pending_action_expired(entry, now_ts=130.0, ttl_seconds=60) is False
    assert is_pending_action_expired(entry, now_ts=161.0, ttl_seconds=60) is True


def test_execute_stored_action_materializes_request_and_uses_runner(tmp_path: Path):
    pending = make_pending_action_entry(
        skill="gentle-cloning",
        actions=_demo_actions(),
    )
    action = _demo_actions()[0]
    captured: dict[str, object] = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        request_path = Path(str(kwargs["input_path"]))
        captured["request_payload"] = json.loads(request_path.read_text(encoding="utf-8"))
        return {"success": True, "output_dir": str(kwargs["output_dir"])}

    result = execute_stored_action(
        pending,
        action,
        runner=fake_runner,
        output_root=tmp_path,
    )

    assert result["success"] is True
    assert captured["skill_name"] == "gentle-cloning"
    assert captured["demo"] is False
    assert captured["timeout"] == 123
    assert captured["request_payload"] == action["request"]
    assert str(captured["output_dir"]).startswith(str(tmp_path / "gentle-cloning_"))


def test_load_bundle_fields_promotes_structured_chat_fields(tmp_path: Path):
    payload = {
        "chat_summary_lines": ["Genome service is missing."],
        "suggested_actions": _demo_actions(),
        "preferred_artifacts": [{"path": "generated/demo.png"}],
    }
    (tmp_path / "result.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (tmp_path / "report.md").write_text("# Fallback report\n", encoding="utf-8")

    fields = load_bundle_fields(tmp_path)

    assert fields["skill_result_json"] == payload
    assert fields["chat_summary_lines"] == ["Genome service is missing."]
    assert fields["suggested_actions"] == _demo_actions()
    assert fields["preferred_artifacts"] == [{"path": "generated/demo.png"}]
    assert fields["report_md"] == "# Fallback report\n"
