"""Tests for clawbio.common.parsers — unified genetic file parsing."""

import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from clawbio.common.parsers import (
    GenotypeRecord,
    detect_format,
    parse_23andme,
    parse_ancestry,
    parse_myheritage,
    parse_vcf,
    parse_vcf_matrix,
    parse_genetic_file,
    genotypes_to_simple,
    genotypes_to_positions,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# GenotypeRecord
# ---------------------------------------------------------------------------

class TestGenotypeRecord:
    def test_to_dict_round_trip(self):
        rec = GenotypeRecord(chrom="1", pos=100, genotype="AG", allele1="A", allele2="G")
        d = rec.to_dict()
        assert d == {"chrom": "1", "pos": 100, "genotype": "AG", "allele1": "A", "allele2": "G"}

    def test_defaults(self):
        rec = GenotypeRecord()
        assert rec.chrom == ""
        assert rec.pos == 0
        assert rec.genotype == ""


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_23andme(self):
        assert detect_format(FIXTURES / "mock_23andme.txt") == "23andme"

    def test_ancestry(self):
        assert detect_format(FIXTURES / "mock_ancestry.txt") == "ancestry"

    def test_vcf(self):
        assert detect_format(FIXTURES / "mock_single_sample.vcf") == "vcf"

    def test_vcf_from_extension(self, tmp_path):
        f = tmp_path / "test.vcf"
        f.write_text("some content\n")
        assert detect_format(f) == "vcf"

    def test_vcf_gz_from_extension(self, tmp_path):
        f = tmp_path / "test.vcf.gz"
        with gzip.open(f, "wt") as fh:
            fh.write("##fileformat=VCFv4.2\n")
        assert detect_format(f) == "vcf"

    def test_myheritage(self):
        assert detect_format(FIXTURES / "mock_myheritage.csv") == "myheritage"

    def test_unknown_raises(self, tmp_path):
        f = tmp_path / "mystery.txt"
        f.write_text("hello world\nno headers here\n")
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            detect_format(f)


# ---------------------------------------------------------------------------
# parse_23andme
# ---------------------------------------------------------------------------

class TestParse23andMe:
    def test_valid_file(self):
        result = parse_23andme(FIXTURES / "mock_23andme.txt")
        assert "rs1234567" in result
        assert result["rs1234567"].genotype == "AG"
        assert result["rs1234567"].chrom == "1"
        assert result["rs1234567"].pos == 100000

    def test_skips_comments(self):
        result = parse_23andme(FIXTURES / "mock_23andme.txt")
        # The comment/header line should not produce a record
        assert not any(k.startswith("#") for k in result)

    def test_handles_i_prefix(self):
        result = parse_23andme(FIXTURES / "mock_23andme.txt")
        assert "i5000001" in result
        assert result["i5000001"].genotype == "GG"

    def test_haploid_genotype(self):
        result = parse_23andme(FIXTURES / "mock_23andme.txt")
        rec = result["rs2222222"]
        assert rec.chrom == "X"
        assert rec.allele1 == "A"
        assert rec.allele2 == ""

    def test_skips_dash_genotype(self, tmp_path):
        f = tmp_path / "dashes.txt"
        f.write_text(
            "# rsid\tchromosome\tposition\tgenotype\n"
            "rs1234567\t1\t100\t--\n"
            "rs7654321\t2\t200\tAG\n"
        )
        result = parse_23andme(f)
        assert "rs1234567" not in result
        assert "rs7654321" in result

    def test_gzip_file(self, tmp_path):
        f = tmp_path / "test.txt.gz"
        with gzip.open(f, "wt") as fh:
            fh.write("# rsid\tchromosome\tposition\tgenotype\n")
            fh.write("rs9999999\t1\t500\tAA\n")
        result = parse_23andme(f)
        assert "rs9999999" in result
        assert result["rs9999999"].genotype == "AA"


# ---------------------------------------------------------------------------
# parse_ancestry
# ---------------------------------------------------------------------------

class TestParseAncestry:
    def test_valid_file(self):
        result = parse_ancestry(FIXTURES / "mock_ancestry.txt")
        assert "rs1234567" in result
        assert result["rs1234567"].genotype == "AG"
        assert result["rs1234567"].allele1 == "A"
        assert result["rs1234567"].allele2 == "G"

    def test_skips_non_rs(self, tmp_path):
        f = tmp_path / "ancestry.txt"
        f.write_text(
            "rsid\tchromosome\tposition\tallele1\tallele2\n"
            "rs111\t1\t100\tA\tG\n"
            "chr1:500\t1\t500\tC\tT\n"
        )
        result = parse_ancestry(f)
        assert "rs111" in result
        assert "chr1:500" not in result

    def test_position_parsing(self):
        result = parse_ancestry(FIXTURES / "mock_ancestry.txt")
        assert result["rs7654321"].pos == 200000


# ---------------------------------------------------------------------------
# parse_myheritage
# ---------------------------------------------------------------------------


class TestParseMyHeritage:
    def test_valid_file(self):
        result = parse_myheritage(FIXTURES / "mock_myheritage.csv")
        assert "rs1234567" in result
        assert result["rs1234567"].genotype == "AG"
        assert result["rs1234567"].allele1 == "A"
        assert result["rs1234567"].allele2 == "G"
        assert result["rs1234567"].chrom == "1"
        assert result["rs1234567"].pos == 100000

    def test_all_variants(self):
        result = parse_myheritage(FIXTURES / "mock_myheritage.csv")
        assert "rs7654321" in result
        assert result["rs7654321"].genotype == "CC"
        assert "rs1111111" in result
        assert result["rs1111111"].genotype == "TT"

    def test_skips_non_rs(self, tmp_path):
        f = tmp_path / "mh.csv"
        f.write_text(
            "RSID,CHROMOSOME,POSITION,RESULT\n"
            "rs111,1,100,AG\n"
            "i5000001,1,200,CC\n"
        )
        result = parse_myheritage(f)
        assert "rs111" in result
        assert "i5000001" not in result

    def test_skips_dash_result(self, tmp_path):
        f = tmp_path / "mh_dash.csv"
        f.write_text(
            "RSID,CHROMOSOME,POSITION,RESULT\n"
            "rs111,1,100,--\n"
            "rs222,1,200,AG\n"
        )
        result = parse_myheritage(f)
        assert "rs111" not in result
        assert "rs222" in result

    def test_lowercase_headers(self, tmp_path):
        f = tmp_path / "mh_lower.csv"
        f.write_text(
            "rsid,chromosome,position,result\n"
            "rs999,1,500,TT\n"
        )
        result = parse_myheritage(f)
        assert "rs999" in result
        assert result["rs999"].genotype == "TT"


# ---------------------------------------------------------------------------
# parse_vcf
# ---------------------------------------------------------------------------

class TestParseVcf:
    def test_valid_file(self):
        result = parse_vcf(FIXTURES / "mock_single_sample.vcf")
        assert "rs1234567" in result
        rec = result["rs1234567"]
        assert rec.allele1 == "A"
        assert rec.allele2 == "G"
        assert rec.genotype == "AG"

    def test_hom_alt(self):
        result = parse_vcf(FIXTURES / "mock_single_sample.vcf")
        rec = result["rs7654321"]
        assert rec.allele1 == "T"
        assert rec.allele2 == "T"
        assert rec.genotype == "TT"

    def test_hom_ref(self):
        result = parse_vcf(FIXTURES / "mock_single_sample.vcf")
        rec = result["rs1111111"]
        assert rec.allele1 == "G"
        assert rec.allele2 == "G"
        assert rec.genotype == "GG"

    def test_phased_genotype(self, tmp_path):
        f = tmp_path / "phased.vcf"
        f.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
            "1\t100\trs999\tA\tG\t50\tPASS\t.\tGT\t0|1\n"
        )
        result = parse_vcf(f)
        assert result["rs999"].genotype == "AG"


# ---------------------------------------------------------------------------
# parse_vcf_matrix
# ---------------------------------------------------------------------------

class TestParseVcfMatrix:
    def test_shape_and_values(self):
        samples, variants, matrix = parse_vcf_matrix(FIXTURES / "mock_single_sample.vcf")
        assert samples == ["SAMPLE1"]
        assert len(variants) == 3
        assert matrix.shape == (1, 3)

    def test_het_hom_encoding(self):
        _, _, matrix = parse_vcf_matrix(FIXTURES / "mock_single_sample.vcf")
        assert matrix[0, 0] == 1   # 0/1 het
        assert matrix[0, 1] == 2   # 1/1 hom alt
        assert matrix[0, 2] == 0   # 0/0 hom ref

    def test_missing_genotype(self, tmp_path):
        f = tmp_path / "missing.vcf"
        f.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
            "1\t100\trs1\tA\tG\t50\tPASS\t.\tGT\t./.\n"
        )
        _, _, matrix = parse_vcf_matrix(f)
        assert matrix[0, 0] == -1

    def test_empty_vcf_raises(self, tmp_path):
        f = tmp_path / "empty.vcf"
        f.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
        )
        with pytest.raises(ValueError, match="No variants"):
            parse_vcf_matrix(f)


# ---------------------------------------------------------------------------
# parse_genetic_file (unified entry point)
# ---------------------------------------------------------------------------

class TestParseGeneticFile:
    def test_auto_detect_23andme(self):
        result = parse_genetic_file(FIXTURES / "mock_23andme.txt")
        assert "rs1234567" in result

    def test_auto_detect_myheritage(self):
        result = parse_genetic_file(FIXTURES / "mock_myheritage.csv")
        assert "rs1234567" in result

    def test_explicit_format(self):
        result = parse_genetic_file(FIXTURES / "mock_ancestry.txt", fmt="ancestry")
        assert "rs1234567" in result

    def test_explicit_myheritage(self):
        result = parse_genetic_file(FIXTURES / "mock_myheritage.csv", fmt="myheritage")
        assert "rs1234567" in result

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown format"):
            parse_genetic_file(FIXTURES / "mock_23andme.txt", fmt="illumina")


# ---------------------------------------------------------------------------
# Convenience converters
# ---------------------------------------------------------------------------

class TestConverters:
    def test_genotypes_to_simple(self):
        records = {"rs1": GenotypeRecord(genotype="AG"), "rs2": GenotypeRecord(genotype="CC")}
        simple = genotypes_to_simple(records)
        assert simple == {"rs1": "AG", "rs2": "CC"}

    def test_genotypes_to_positions(self):
        records = {"rs1": GenotypeRecord(chrom="1", pos=100)}
        positions = genotypes_to_positions(records)
        assert positions == {"rs1": {"chrom": "1", "pos": 100}}
