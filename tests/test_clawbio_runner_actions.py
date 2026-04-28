from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_RUNNER_SPEC = importlib.util.spec_from_file_location("clawbio_runner", PROJECT_ROOT / "clawbio.py")
assert _RUNNER_SPEC and _RUNNER_SPEC.loader
clawbio_runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(clawbio_runner)


def test_run_skill_promotes_structured_result_fields(monkeypatch, tmp_path: Path):
    fake_script = tmp_path / "gentle_cloning.py"
    fake_script.write_text("# placeholder\n", encoding="utf-8")

    monkeypatch.setitem(
        clawbio_runner.SKILLS,
        "gentle-cloning",
        {
            "script": fake_script,
            "demo_args": ["--demo"],
            "description": "GENtle wrapper",
            "allowed_extra_flags": set(),
        },
    )

    class Proc:
        returncode = 0
        stdout = "wrapper ok\n"
        stderr = ""

    def fake_run(cmd, capture_output, text, timeout, cwd):
        output_dir = Path(cmd[cmd.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.md").write_text("# Fallback report\n", encoding="utf-8")
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "schema": "gentle.clawbio_skill_result.v1",
                    "chat_summary_lines": [
                        "GENtle prepared an isoform guide response.",
                    ],
                    "preferred_artifacts": ["report.md", "result.json"],
                    "suggested_actions": [
                        {
                            "action_id": "show-bach2-isoforms",
                            "label": "Show BACH2 isoform guide",
                            "kind": "navigation",
                            "request": {
                                "schema": "gentle.clawbio_skill_request.v1",
                                "mode": "raw",
                                "raw_args": [
                                    "services",
                                    "guide",
                                    "--channel",
                                    "telegram",
                                    "--section",
                                    "isoforms",
                                    "--gene",
                                    "BACH2",
                                ],
                            },
                            "requires_confirmation": False,
                        }
                    ],
                    "report_md": "# Structured Report\n",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return Proc()

    monkeypatch.setattr(clawbio_runner.subprocess, "run", fake_run)

    output_dir = tmp_path / "runner_out"
    result = clawbio_runner.run_skill(
        skill_name="gentle-cloning",
        demo=True,
        output_dir=str(output_dir),
    )

    assert result["success"] is True
    assert result["stdout"] == "wrapper ok\n"
    assert result["result_json_path"] == str(output_dir / "result.json")
    assert result["report_md"] == "# Structured Report\n"
    assert result["chat_summary_lines"] == ["GENtle prepared an isoform guide response."]
    assert result["preferred_artifacts"] == ["report.md", "result.json"]
    assert result["suggested_actions"][0]["action_id"] == "show-bach2-isoforms"
    assert result["skill_result_json"]["schema"] == "gentle.clawbio_skill_result.v1"
