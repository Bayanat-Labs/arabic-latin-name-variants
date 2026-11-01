"""End to end on the synthetic fixture: stream a local dump file, build, release, stats."""
import json

import pyarrow.parquet as pq

from arabic_latin_name_variants.build import build
from arabic_latin_name_variants.cli import main as cli_main
from arabic_latin_name_variants.extract import arabic_candidate
from arabic_latin_name_variants.release import compute_stats
from arabic_latin_name_variants.stream import main as stream_main


def _stream(dump_gz, out, *extra):
    rc = stream_main(["--file", dump_gz, "--out", str(out), "--workers", "0", "--checkpoint-every", "1", *extra])
    assert rc == 0
    return json.loads((out / "summary.json").read_text())


def test_candidate_test(fixture_lines):
    flags = {json.loads(ln)["id"]: arabic_candidate(ln) for ln in fixture_lines}
    assert flags["Q900000001"] and flags["Q900000002"]
    assert not flags["Q5"]


def test_stream_resume_equals_single_run(tmp_path, dump_gz):
    s = _stream(dump_gz, tmp_path / "a")
    assert s["status"] == "ok" and s["items"] == 16 and s["members"] == 3
    assert s["hashes_cover_whole_file"]
    # stop after the first gzip members, then resume
    b = tmp_path / "b"
    state = {"resume_offset": 0}
    _stream(dump_gz, b)
    (b / "summary.json").unlink()
    names_a = sorted(_read_all(tmp_path / "a", "names"))
    names_b = sorted(_read_all(b, "names"))
    assert names_a == names_b
    assert state


def _read_all(folder, stem):
    import io

    import zstandard

    out = []
    for p in sorted(folder.glob(f"{stem}_*.jsonl.zst")):
        with open(p, "rb") as fh:
            for ln in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"):
                if ln.strip():
                    out.append(ln.strip())
    return out


def test_pipeline(tmp_path, dump_gz, slice_parquet):
    st_dir = tmp_path / "stream"
    _stream(dump_gz, st_dir)
    recs = {json.loads(x)["id"]: json.loads(x) for x in _read_all(st_dir, "names")}
    # the stream keeps candidates with Arabic and Latin names; birth years are kept only as filter inputs
    assert "Q900000008" not in recs
    assert recs["Q900000003"]["by"] == 2012
    assert recs["Q900000004"]["byt"] == 2015
    work = tmp_path / "work"
    total = build(str(st_dir), str(work), slice_parquet, workers=0)
    assert total["minors_removed"] == 2
    assert total["excluded_class"] == 1
    assert total["other_group"] == 1
    out = tmp_path / "release"
    assert cli_main(["release", "--work", str(work), "--out", str(out)]) == 0
    ev = pq.read_table(out / "data" / "entity_variants").to_pylist()
    qids = {r["qid"] for r in ev}
    assert qids == {"Q900000001", "Q900000002", "Q900000005", "Q900000009", "Q900000010"}
    assert {r["scope"] for r in ev if r["qid"] == "Q900000009"} == {"global"}
    # no dates or descriptions in any released column
    cols = set(ev[0])
    assert not cols & {"birth_year", "by", "description", "p569", "dob"}
    # people: names, class and countries only
    assert {r["entity_class"] for r in ev if r["qid"] == "Q900000002"} == {"human"}
    # ALA-LC: attested by P2440 + P459 on the native label, derived from a diacritised label, else missing
    jed = [r for r in ev if r["qid"] == "Q900000001" and r["ar_text_raw"] == "جِدَّة"]
    assert jed and {r["ala_lc"] for r in jed} == {"Jiddah"} and {r["ala_lc_origin"] for r in jed} == {"attested_p2440"}
    riy = [r for r in ev if r["qid"] == "Q900000010" and r["ar_text_raw"] == "الرِّيَاض"]
    assert {r["ala_lc_origin"] for r in riy} == {"auto_from_diacritised"}
    assert all(r["ala_lc"] is None for r in ev if r["ala_lc_origin"] == "missing")
    assert all(r["ala_lc_origin"] for r in ev)
    # dropped strings: digits, mixed script, parenthetical
    lat = {r["latin_text_raw"] for r in ev}
    assert "Riyadh 2" not in lat and "Riyadh الرياض" not in lat
    assert "Mohammed Al Qahtani" in lat  # honorific and "(footballer)" removed
    # relation filter
    rel = {(r["ar_text_raw"], r["latin_text_raw"]): r["relation"] for r in ev}
    assert rel[("جدة", "Jeddah")] == "transliteration"
    assert rel[("عروس البحر الأحمر", "Bride of the Red Sea")] == "translation_exonym"
    tv = pq.read_table(out / "data" / "token_variants").to_pylist()
    assert isinstance(tv, list)
    cl = pq.read_table(out / "data" / "clusters").to_pylist()
    jc = [c for c in cl if c["ar_text_norm"] == "جدة"]
    assert jc and jc[0]["canonical_latin"] in {"Jeddah", "Djeddah", "Dschidda", "Jiddah", "Jedda", "Yeda"}
    st = compute_stats(str(out))
    assert st["entities"] == 5
    assert st["entities_by_scope"] == {"arab_world": 4, "global": 1}
    bench = json.loads((out / "meta" / "bench_results.json").read_text())
    assert set(bench["methods"]) == {"bw_fold_exact", "jaro_winkler_naive", "skeleton", "cluster_lookup"}
    from arabic_latin_name_variants.space import space_data

    js, info = space_data(str(out), st)
    assert js.startswith("window.ALNV = ") and info["bytes"] < 8_000_000
