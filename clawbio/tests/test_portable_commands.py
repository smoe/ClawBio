"""
Tests for clawbio.common.portable_commands
Run with: pytest tests/test_portable_commands.py -v
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from clawbio.common.portable_commands import (
    build_portable_commands_sh,
    write_portable_commands_sh,
)


# ── build_portable_commands_sh tests ──────────────────────────────────────────

class TestBuildPortableCommandsSh:

    def test_returns_string(self):
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "CRISPR", "--output": "./report"},
        )
        assert isinstance(result, str)

    def test_contains_shebang(self):
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "CRISPR"},
        )
        assert result.startswith("#!/usr/bin/env bash")

    def test_contains_repo_root_anchor(self):
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "CRISPR"},
        )
        assert "BASH_SOURCE" in result
        assert "REPO_ROOT" in result
        assert 'skills' in result

    def test_no_absolute_paths_in_output(self):
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "CRISPR", "--output": "./report"},
        )
        # output should be relative ./report not absolute
        assert "./report" in result

    def test_skill_name_in_script_path(self):
        result = build_portable_commands_sh(
            skill_name="vcf-annotator",
            script_name="vcf_annotator.py",
            args={"--demo": None},
        )
        assert "vcf-annotator/vcf_annotator.py" in result

    def test_boolean_flag_no_value(self):
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--demo": None},
        )
        assert "--demo" in result
        # Should not have "--demo None"
        assert "--demo None" not in result

    def test_custom_generated_at(self):
        ts = "2026-04-19 10:00 UTC"
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "test"},
            generated_at=ts,
        )
        assert ts in result

    def test_default_generated_at_present(self):
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "test"},
        )
        # Should contain a date
        assert "202" in result  # year 202x

    def test_multiple_args(self):
        result = build_portable_commands_sh(
            skill_name="vcf-annotator",
            script_name="vcf_annotator.py",
            args={
                "--input": "variants.vcf",
                "--output": "./report",
            },
        )
        assert "--input" in result
        assert "--output" in result
        assert "variants.vcf" in result

    def test_no_machine_specific_paths(self):
        """Command must not embed /home/... or C:\\ style absolute paths."""
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--output": "./report"},
        )
        assert "/home/" not in result
        assert "C:\\" not in result
        assert "/Users/" not in result

    def test_uses_dollar_repo_root_variable(self):
        """Script path must use $REPO_ROOT variable, not hardcoded path."""
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "test"},
        )
        assert '"$REPO_ROOT/skills/' in result

    def test_set_euo_pipefail(self):
        """Script should fail fast on errors."""
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "test"},
        )
        assert "set -euo pipefail" in result

    def test_error_if_skills_not_found(self):
        """Script should exit with error if repo root not found."""
        result = build_portable_commands_sh(
            skill_name="lit-synthesizer",
            script_name="lit_synthesizer.py",
            args={"--query": "test"},
        )
        assert "ERROR" in result
        assert "exit 1" in result


# ── write_portable_commands_sh tests ──────────────────────────────────────────

class TestWritePortableCommandsSh:

    def test_creates_commands_sh(self):
        with tempfile.TemporaryDirectory() as tmp:
            repro = Path(tmp) / "reproducibility"
            write_portable_commands_sh(
                repro_dir=repro,
                skill_name="lit-synthesizer",
                script_name="lit_synthesizer.py",
                args={"--query": "CRISPR", "--output": "./report"},
            )
            assert (repro / "commands.sh").exists()

    def test_creates_repro_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            repro = Path(tmp) / "new" / "reproducibility"
            assert not repro.exists()
            write_portable_commands_sh(
                repro_dir=repro,
                skill_name="lit-synthesizer",
                script_name="lit_synthesizer.py",
                args={"--query": "test"},
            )
            assert repro.exists()
            assert (repro / "commands.sh").exists()

    def test_content_is_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repro = Path(tmp) / "reproducibility"
            write_portable_commands_sh(
                repro_dir=repro,
                skill_name="vcf-annotator",
                script_name="vcf_annotator.py",
                args={"--input": "variants.vcf", "--output": "./report"},
            )
            content = (repro / "commands.sh").read_text()
            assert "REPO_ROOT" in content
            assert "BASH_SOURCE" in content
            assert "/home/" not in content

    def test_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            repro = Path(tmp) / "reproducibility"
            repro.mkdir()
            (repro / "commands.sh").write_text("old content")
            write_portable_commands_sh(
                repro_dir=repro,
                skill_name="lit-synthesizer",
                script_name="lit_synthesizer.py",
                args={"--query": "new"},
            )
            content = (repro / "commands.sh").read_text()
            assert "old content" not in content
            assert "new" in content


# ── Integration: lit-synthesizer generates portable commands ──────────────────

class TestLitSynthesizerPortableCommands:
    """Test that lit_synthesizer.py generates portable commands.sh"""

    def test_commands_sh_uses_repo_root(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                               "lit-synthesizer" / "skills" / "lit-synthesizer"))
        try:
            from lit_synthesizer import DEMO_PAPERS, build_citation_graph, generate_report
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                graph = build_citation_graph(DEMO_PAPERS)
                generate_report("CRISPR", DEMO_PAPERS, graph, out)
                content = (out / "reproducibility" / "commands.sh").read_text()
                assert "REPO_ROOT" in content
                assert "BASH_SOURCE" in content
                assert str(out) not in content  # no absolute output path
        except ImportError:
            pytest.skip("lit_synthesizer not available in this test run")


# ── Integration: vcf-annotator generates portable commands ───────────────────

class TestVCFAnnotatorPortableCommands:
    """Test that vcf_annotator.py generates portable commands.sh"""

    def test_commands_sh_uses_repo_root(self):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                               "vcf-annotator" / "skills" / "vcf-annotator"))
        try:
            from vcf_annotator import DEMO_ANNOTATIONS, generate_report
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                generate_report(DEMO_ANNOTATIONS, out, "demo.vcf")
                content = (out / "reproducibility" / "commands.sh").read_text()
                assert "REPO_ROOT" in content
                assert "BASH_SOURCE" in content
                assert str(out) not in content  # no absolute output path
        except ImportError:
            pytest.skip("vcf_annotator not available in this test run")
