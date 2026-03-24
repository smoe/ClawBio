"""Tests for clawbio.common.profile — PatientProfile persistence and access."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from clawbio.common.profile import PatientProfile
from clawbio.common.parsers import GenotypeRecord

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestPatientProfileInit:
    def test_defaults(self):
        p = PatientProfile()
        assert p.metadata["patient_id"] == ""
        assert p.genotype_count == 0
        assert p.skill_results == {}
        assert p.metadata["upload_date"]  # auto-populated

    def test_explicit_fields(self):
        p = PatientProfile(
            patient_id="demo",
            input_file="/tmp/test.txt",
            checksum="abc123",
        )
        assert p.metadata["patient_id"] == "demo"
        assert p.metadata["checksum"] == "abc123"


class TestFromGeneticFile:
    def test_creates_profile(self):
        p = PatientProfile.from_genetic_file(FIXTURES / "mock_23andme.txt", patient_id="test")
        assert p.metadata["patient_id"] == "test"
        assert p.genotype_count == 5  # 4 rs + 1 i-prefix
        assert p.metadata["checksum"]  # non-empty

    def test_auto_patient_id(self):
        p = PatientProfile.from_genetic_file(FIXTURES / "mock_23andme.txt")
        assert p.metadata["patient_id"] == "mock_23andme"


# ---------------------------------------------------------------------------
# Genotype access
# ---------------------------------------------------------------------------

class TestGenotypeAccess:
    @pytest.fixture
    def profile(self):
        return PatientProfile.from_genetic_file(FIXTURES / "mock_23andme.txt")

    def test_get_genotypes_all(self, profile):
        genos = profile.get_genotypes()
        assert len(genos) == 5
        assert genos["rs1234567"] == "AG"

    def test_get_genotypes_filtered(self, profile):
        genos = profile.get_genotypes(rsids=["rs1234567", "rs_missing"])
        assert "rs1234567" in genos
        assert "rs_missing" not in genos

    def test_get_records(self, profile):
        records = profile.get_records(rsids=["rs1234567"])
        assert isinstance(records["rs1234567"], GenotypeRecord)
        assert records["rs1234567"].genotype == "AG"

    def test_genotype_count(self, profile):
        assert profile.genotype_count == 5


# ---------------------------------------------------------------------------
# Skill results
# ---------------------------------------------------------------------------

class TestSkillResults:
    def test_add_and_get(self):
        p = PatientProfile(patient_id="test")
        p.add_skill_result("pharmgx", {"drug": "codeine", "action": "avoid"})
        result = p.get_skill_result("pharmgx")
        assert result == {"drug": "codeine", "action": "avoid"}

    def test_missing_skill_returns_none(self):
        p = PatientProfile()
        assert p.get_skill_result("nonexistent") is None


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        p = PatientProfile.from_genetic_file(FIXTURES / "mock_23andme.txt", patient_id="rt")
        p.add_skill_result("test_skill", {"score": 42})

        path = p.save(tmp_path / "profile.json")
        assert path.exists()

        loaded = PatientProfile.load(path)
        assert loaded.metadata["patient_id"] == "rt"
        assert loaded.genotype_count == 5
        assert loaded.get_skill_result("test_skill") == {"score": 42}

    def test_save_creates_parent_dirs(self, tmp_path):
        p = PatientProfile(patient_id="x")
        path = p.save(tmp_path / "nested" / "dir" / "profile.json")
        assert path.exists()

    def test_json_is_valid(self, tmp_path):
        p = PatientProfile(patient_id="json_test")
        path = p.save(tmp_path / "p.json")
        data = json.loads(path.read_text())
        assert "metadata" in data
        assert "genotypes" in data
        assert "skill_results" in data


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_contains_id_and_count(self):
        p = PatientProfile(patient_id="demo")
        r = repr(p)
        assert "demo" in r
        assert "genotypes=0" in r
