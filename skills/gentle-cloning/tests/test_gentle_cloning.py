from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _skill_script() -> Path:
    return Path(__file__).resolve().parents[1] / "gentle_cloning.py"


def test_demo_writes_expected_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo_out"
    cmd = [
        sys.executable,
        str(_skill_script()),
        "--demo",
        "--output",
        str(output_dir),
        "--gentle-cli",
        "true",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr

    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "gentle.clawbio_skill_result.v1"
    assert payload["status"] == "ok"
    assert payload["request"]["raw_args"] == ["capabilities"]
    assert (output_dir / "report.md").exists()


def test_wrapper_preserves_structured_stdout_fields(tmp_path: Path) -> None:
    fake_cli = tmp_path / "fake_gentle_cli.py"
    fake_cli.write_text(
        "\n".join(
            [
                "import json",
                "print(json.dumps({",
                "  'chat_summary_lines': ['Guide ready.'],",
                "  'suggested_actions': [{",
                "    'action_id': 'show-bach2',",
                "    'label': 'Show BACH2 isoform guide',",
                "    'request': {'schema': 'gentle.clawbio_skill_request.v1', 'mode': 'raw', 'raw_args': ['services', 'guide']},",
                "    'requires_confirmation': False",
                "  }]",
                "}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
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
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(_skill_script()),
        "--input",
        str(request),
        "--output",
        str(output_dir),
        "--gentle-cli",
        f"{sys.executable} {fake_cli}",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr

    payload = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["chat_summary_lines"] == ["Guide ready."]
    assert payload["suggested_actions"][0]["action_id"] == "show-bach2"
