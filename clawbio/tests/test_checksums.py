"""Tests for clawbio.common.checksums — SHA-256 helpers."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from clawbio.common.checksums import sha256_file, sha256_hex


class TestSha256File:
    def test_known_content(self, tmp_path):
        f = tmp_path / "known.txt"
        content = b"hello clawbio\n"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert sha256_file(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert sha256_file(f) == expected


class TestSha256Hex:
    def test_truncation(self, tmp_path):
        f = tmp_path / "trunc.txt"
        f.write_bytes(b"test")
        full = sha256_file(f)
        assert sha256_hex(f, length=16) == full[:16]
        assert len(sha256_hex(f, length=8)) == 8
