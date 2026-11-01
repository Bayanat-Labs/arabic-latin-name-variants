import gzip
import os

import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def fixture_lines():
    with open(os.path.join(FIX, "entities.jsonl"), "rb") as fh:
        return [ln.rstrip(b"\n") for ln in fh if ln.strip()]


@pytest.fixture
def dump_gz(tmp_path, fixture_lines):
    """The fixture entities as a small multi-member gzip file laid out like the Wikidata dump."""
    path = tmp_path / "wikidata-test-all.json.gz"
    lines = fixture_lines
    half = len(lines) // 2
    parts = [b"[\n", b",\n".join(lines[:half]) + b",\n", b",\n".join(lines[half:]) + b"\n]\n"]
    with open(path, "wb") as fh:
        for p in parts:
            fh.write(gzip.compress(p, mtime=0))
    return str(path)


@pytest.fixture
def slice_parquet(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.table({"qid": ["Q900000001", "Q900000002", "Q900000005", "Q900000010"],
                  "countries": [["Q851"], ["Q851"], ["Q851"], ["Q851"]]})
    path = tmp_path / "slice.parquet"
    pq.write_table(t, path)
    return str(path)
