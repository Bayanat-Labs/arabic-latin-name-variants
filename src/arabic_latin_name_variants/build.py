"""Build the entity_variants rows from the stream's name records.

Input: the stream folder (names_*.jsonl.zst, classes_*.jsonl.zst) and the Arab-World Wikidata
slice entities table (its QIDs decide the arab_world scope; its countries are used for them).
Output (work folder): entity_variants parts and per-part stats; then ``aggregate.py`` turns these
into the release tables.
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from .alalc import ala_lc_style, romanise
from .align import align_full, aligned_words, arabic_units, style_tag, unmatched_word, word_coverage
from .config import (ALA_LC, ARABIC_LANGS, DIN_31635, DUMP_DATE, KEPT_GROUPS, LATIN_LANGS, MIN_AGE_YEARS,
                     NAME_PROPS, SAME_NAME_BIRTH_YEAR)
from .extract import latest_birth_year
from .skeleton import relation, similarity
from .text import (buckwalter, clean, human_string_reason, is_diacritised, latin_fold, normalize_ar, reorder_inverted,
                   string_sha256)

S = pa.string()
F32 = pa.float32()
LS = pa.list_(S)
ALIGN_T = pa.list_(pa.struct([("ar_token", S), ("latin_token", S), ("score", F32)]))
VARIANT_SCHEMA = pa.schema([
    ("qid", S), ("scope", S), ("entity_class", S), ("countries", LS),
    ("ar_text_raw", S), ("ar_text_norm", S), ("ar_langs", LS), ("ar_kinds", LS),
    ("buckwalter", S), ("ala_lc", S), ("ala_lc_origin", S), ("din_31635", S),
    ("latin_text_raw", S), ("latin_fold", S), ("latin_langs", LS), ("latin_kinds", LS), ("method_qids", LS),
    ("relation", S), ("skeleton_sim", F32), ("token_alignment", ALIGN_T), ("alignment_score", F32),
    ("lastrevid", pa.int64()),
])
ENTITY_SCHEMA = pa.schema([
    ("qid", S), ("scope", S), ("entity_class", S), ("countries", LS), ("n_ar", pa.int32()),
    ("n_latin", pa.int32()), ("n_pairs", pa.int32()), ("lastrevid", pa.int64()),
])
AR_LANG_SET = frozenset(ARABIC_LANGS)
AR_OR_MUL = frozenset(ARABIC_LANGS) | {"mul"}
LAT_SET = frozenset(LATIN_LANGS)


def iter_records(path: str):
    import zstandard

    with open(path, "rb") as fh:
        reader = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(fh), buffer_size=1 << 20)
        for line in reader:
            if line.strip():
                yield json.loads(line)


def _add(d: dict, text: str, lang: str | None, kind: str, methods=()):
    e = d.setdefault(text, {"langs": set(), "kinds": set(), "methods": set(), "p2440": []})
    if lang:
        e["langs"].add(lang)
    e["kinds"].add(kind)
    e["methods"].update(methods)
    return e


def load_drop_list(path: str | None) -> frozenset:
    """(qid, sha256) pairs from a drop list file, one `{qid: Q1, sha256: ...}` entry per line, so no YAML
    library is needed."""
    if not path:
        return frozenset()
    import re

    pat = re.compile(r"\{qid: (Q\d+), sha256: ([0-9a-f]{64})\}")
    with open(path, encoding="utf-8") as fh:
        return frozenset(m.groups() for m in map(pat.search, fh) if m)


def entity_strings(rec: dict, person: bool, drops: dict, drop_list: frozenset = frozenset()) -> tuple[dict, dict, bool]:
    """(Arabic strings, Latin strings, has an ar/arz/ary Arabic string) after cleaning. A person's
    strings also pass the non-name filter and the drop list."""
    ar: dict = {}
    la: dict = {}
    qualifies = False

    def take(value: str, lang: str | None, kind: str, methods=()):
        nonlocal qualifies
        text, script = clean(value, person=person)
        if text is None:
            drops[script] = drops.get(script, 0) + 1
            return None, None
        if person:
            if human_string_reason(text, script):
                drops["non_name"] = drops.get("non_name", 0) + 1
                return None, None
            if (rec["id"], string_sha256(text, script)) in drop_list:
                drops["drop_list"] = drops.get("drop_list", 0) + 1
                return None, None
        if script == "arabic":
            if lang in AR_OR_MUL:
                if lang in AR_LANG_SET:
                    qualifies = True
                _add(ar, text, lang, kind)
                return "arabic", text
            drops["arabic_in_latin_lang"] = drops.get("arabic_in_latin_lang", 0) + 1
            return None, None
        if lang is None or lang in LAT_SET:
            _add(la, text, lang, kind, methods)
            return "latin", text
        drops["latin_in_arabic_lang"] = drops.get("latin_in_arabic_lang", 0) + 1
        return None, None

    for lang, kind, value in rec.get("t", []):
        take(value, lang, kind)
    for pid, lang, value, _rank, translits, methods in rec.get("m", []):
        script, text = take(value, lang, NAME_PROPS[pid])
        for tl in translits:
            tscript, ttext = take(tl, None, "p2440_transliteration", methods)
            if script == "arabic" and tscript == "latin" and text in ar:
                ar[text]["p2440"].append((ttext, tuple(methods)))
    for value, methods, _rank in rec.get("x", []):
        take(value, None, "p2440_transliteration", methods)
    return ar, la, qualifies


def choose_ala_lc(ar_text: str, attested: str | None, din: str | None, latin) -> tuple[str | None, str, str | None]:
    """(ala_lc, ala_lc_origin, din_31635) of an Arabic name, given its P2440 values and the Latin
    names kept for it; a value that is not among those names is not used."""
    latin = set(latin)
    din = din if din in latin else None
    if attested in latin:
        return attested, "attested_p2440", din
    best, best_sim = None, 0.0
    for lt in sorted(latin):
        if ala_lc_style(lt):
            s = similarity(ar_text, lt)
            if s >= 0.8 and s > best_sim:
                best, best_sim = lt, s
    if best:
        return best, "attested_variant", din
    if is_diacritised(ar_text):
        r = romanise(ar_text)
        if r:
            return r, "auto_from_diacritised", din
    return None, "missing", din


def may_be_alive(latest: int | None) -> bool:
    """The same-name rule's test: a latest possible birth year of 1900 or later, or none known."""
    return latest is None or latest >= SAME_NAME_BIRTH_YEAR


def same_name_reasons(rows: list[dict]) -> list[str | None]:
    """For one person who may be alive: why each name pair does not read as the same name, or None.
    A translation/exonym pair goes. So does a pair with a word of three or more letters on either side
    that is neither a listed particle or title nor matched in a word pair of any row of the entity."""
    ar_all: set[str] = set()
    la_all: set[str] = set()
    for r in rows:
        a, l = aligned_words(r["token_alignment"])
        ar_all |= a
        la_all |= l
    out: list[str | None] = []
    for r in rows:
        if r["relation"] == "translation_exonym":
            out.append("translation_exonym")
        elif unmatched_word(r["ar_text_raw"], r["latin_text_raw"], ar_all, la_all):
            out.append("unmatched_word")
        else:
            out.append(None)
    return out


def process_file(path: str, work: str, slice_path: str, class_table: str, scope_only: str | None = None,
                 stem: str | None = None, drop_list_path: str | None = None) -> dict:
    """Turn one names file into Parquet parts. Returns stats. With scope_only ("arab_world" or "global"),
    records of the other scope are skipped (counted as other_scope). drop_list_path: the reviewed list
    of (qid, sha256) strings that are not used for people."""
    from .classes import CompactClassifier

    clf = _CLF.get(class_table)
    if clf is None:
        with open(class_table, encoding="utf-8") as fh:
            clf = _CLF[class_table] = CompactClassifier(json.load(fh))
    key = (slice_path, scope_only == "global")
    sl = _SLICE.get(key)
    if sl is None:
        sl = _SLICE[key] = load_slice(slice_path, qids_only=scope_only == "global")
    cutoff = int(DUMP_DATE[:4]) - MIN_AGE_YEARS
    drop_list = load_drop_list(drop_list_path)
    stem = stem or os.path.basename(path).split(".")[0]
    for sub in ("variants/arab_world", "variants/global", "entities"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    writers = {
        "arab_world": pq.ParquetWriter(os.path.join(work, "variants", "arab_world", stem + ".parquet"),
                                       VARIANT_SCHEMA, compression="zstd", compression_level=9),
        "global": pq.ParquetWriter(os.path.join(work, "variants", "global", stem + ".parquet"), VARIANT_SCHEMA,
                                   compression="zstd", compression_level=9),
        "entities": pq.ParquetWriter(os.path.join(work, "entities", stem + ".parquet"), ENTITY_SCHEMA,
                                     compression="zstd"),
    }
    schemas = {"arab_world": VARIANT_SCHEMA, "global": VARIANT_SCHEMA, "entities": ENTITY_SCHEMA}
    rows_aw: list = []
    rows_gl: list = []
    ents: list = []

    def flush(force: bool = False) -> None:
        for name, buf in (("arab_world", rows_aw), ("global", rows_gl), ("entities", ents)):
            if buf and (force or len(buf) >= 10_000):
                writers[name].write_table(pa.Table.from_pylist(buf, schema=schemas[name]))
                buf.clear()
    st = {"records": 0, "excluded_class": 0, "other_group": 0, "minors_removed": 0, "no_arabic": 0,
          "no_latin": 0, "entities": 0, "pairs": 0, "drops": {}, "repeated": 0, "other_scope": 0,
          "same_name_translation_exonym": 0, "same_name_unmatched_word": 0, "no_same_name_pair": 0}
    seen = set()
    for rec in iter_records(path):
        st["records"] += 1
        q = rec["id"]
        if q in seen:
            st["repeated"] += 1
            continue
        seen.add(q)
        if scope_only and (q in sl) != (scope_only == "arab_world"):
            st["other_scope"] += 1
            continue
        p31 = rec["p31"]
        if clf.is_excluded(p31):
            st["excluded_class"] += 1
            continue
        person = clf.is_person(p31)
        group = "human" if person else clf.group(p31)
        if group not in KEPT_GROUPS:
            st["other_group"] += 1
            continue
        latest = None
        if person:
            latest = latest_birth_year(rec.get("by"), rec.get("byt"), rec.get("cyt"))
            if latest is not None and latest >= cutoff:
                st["minors_removed"] += 1
                continue
        ar, la, qualifies = entity_strings(rec, person, st["drops"], drop_list)
        if not qualifies:
            st["no_arabic"] += 1
            continue
        if not la:
            st["no_latin"] += 1
            continue
        pair_rows = []
        for a in sorted(ar):
            for lt in sorted(la):
                sim = similarity(a, lt)
                cmp_text = lt
                inv = reorder_inverted(lt)
                if inv:
                    s2 = similarity(a, inv)
                    if s2 > sim:
                        sim, cmp_text = s2, inv
                sim = round(sim, 4)
                rel = relation(sim)
                if rel == "translation_exonym" and word_coverage(a, cmp_text):
                    rel = "mixed"
                pairs, ascore = ([], None)
                if rel != "translation_exonym":
                    pairs, ascore, skips = align_full(a, cmp_text)
                    if rel == "transliteration" and skips:
                        rel = "mixed"
                pair_rows.append({"ar_text_raw": a, "latin_text_raw": lt, "relation": rel, "skeleton_sim": sim,
                                  "token_alignment": [{"ar_token": x, "latin_token": y, "score": s} for x, y, s in pairs],
                                  "alignment_score": ascore})
        if person and may_be_alive(latest):
            kept = []
            for row, why in zip(pair_rows, same_name_reasons(pair_rows)):
                if why:
                    st["same_name_" + why] += 1
                else:
                    kept.append(row)
            pair_rows = kept
            if not pair_rows:
                st["no_same_name_pair"] += 1
                continue
        if q in sl:
            scope, countries = "arab_world", list(sl[q])
        else:
            scope = "global"
            countries = list(dict.fromkeys((rec.get("p27") or []) + (rec.get("p17") or []))) if person \
                else list(rec.get("p17") or [])
        st["entities"] += 1
        rows = rows_aw if scope == "arab_world" else rows_gl
        latin_for: dict[str, list] = {}
        for row in pair_rows:
            latin_for.setdefault(row["ar_text_raw"], []).append(row["latin_text_raw"])
        fields = {}
        for a, names in latin_for.items():
            info = ar[a]
            din = next((t for t, ms in info["p2440"] if DIN_31635 in ms), None)
            att = next((t for t, ms in info["p2440"] if ALA_LC in ms), None)
            fields[a] = (normalize_ar(a), buckwalter(a)) + choose_ala_lc(a, att, din, names)
        for row in pair_rows:
            a, lt = row["ar_text_raw"], row["latin_text_raw"]
            a_norm, bw, ala, origin, din = fields[a]
            linfo = la[lt]
            rows.append({
                "qid": q, "scope": scope, "entity_class": group, "countries": countries,
                "ar_text_raw": a, "ar_text_norm": a_norm, "ar_langs": sorted(ar[a]["langs"]),
                "ar_kinds": sorted(ar[a]["kinds"]), "buckwalter": bw, "ala_lc": ala,
                "ala_lc_origin": origin, "din_31635": din,
                "latin_text_raw": lt, "latin_fold": latin_fold(lt), "latin_langs": sorted(linfo["langs"]),
                "latin_kinds": sorted(linfo["kinds"]), "method_qids": sorted(linfo["methods"]),
                "relation": row["relation"], "skeleton_sim": row["skeleton_sim"],
                "token_alignment": row["token_alignment"],
                "alignment_score": row["alignment_score"], "lastrevid": rec.get("rev"),
            })
        st["pairs"] += len(pair_rows)
        ents.append({"qid": q, "scope": scope, "entity_class": group, "countries": countries,
                     "n_ar": len(latin_for), "n_latin": len({r["latin_text_raw"] for r in pair_rows}),
                     "n_pairs": len(pair_rows), "lastrevid": rec.get("rev")})
        flush()
    flush(force=True)
    for w in writers.values():
        w.close()
    st["file"] = os.path.basename(path)
    return st


_CLF: dict = {}
_SLICE: dict = {}


def load_slice(path: str, qids_only: bool = False):
    """QID -> countries from the Arab-World Wikidata slice entities table (file or folder); with
    qids_only, just the set of QIDs."""
    files = sorted(glob.glob(os.path.join(path, "*.parquet"))) if os.path.isdir(path) else [path]
    if qids_only:
        return frozenset(q for f in files for q in pq.read_table(f, columns=["qid"]).column("qid").to_pylist())
    out: dict[str, tuple] = {}
    for f in files:
        t = pq.read_table(f, columns=["qid", "countries"])
        for q, c in zip(t.column("qid").to_pylist(), t.column("countries").to_pylist()):
            out[q] = tuple(sys.intern(x) for x in (c or ()))
    return out


def write_class_table(stream_dir: str, out_path: str) -> dict:
    from .classes import Classifier, compact_table, load_parents

    table = compact_table(Classifier(load_parents(stream_dir)))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(table, fh, separators=(",", ":"), sort_keys=True)
    return {"classes_with_p279": table["n_classes"], "classes_in_groups": len(table["dist"]),
            "person_classes": len(table["person"]), "excluded_classes": len(table["excluded"])}


def build(stream_dir: str, work: str, slice_path: str, workers: int = 4, files: list[str] | None = None,
          min_free_mb: int = 0, class_table_from: str | None = None, drop_list_path: str | None = None) -> dict:
    """Process every names file. With min_free_mb, a file is started only while the disk holding the
    work folder has at least that much free space. The class table comes from the stream folder's
    classes_*.jsonl.zst files, or from class_table_from (a folder of such files from the same dump).
    drop_list_path: an optional drop list file, see load_drop_list."""
    import shutil
    import time

    files = files or sorted(glob.glob(os.path.join(stream_dir, "names_*.jsonl.zst")))
    os.makedirs(work, exist_ok=True)
    class_table = os.path.join(work, "class_table.json")
    ct = write_class_table(class_table_from or stream_dir, class_table)
    stats = []

    def wait_for_space() -> None:
        while min_free_mb and shutil.disk_usage(work).free < min_free_mb * 1024 * 1024:
            time.sleep(15)

    if workers <= 0:
        for f in files:
            wait_for_space()
            stats.append(process_file(f, work, slice_path, class_table, drop_list_path=drop_list_path))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            pending = []
            for f in files:
                while len(pending) >= workers:
                    stats.append(pending.pop(0).result())
                wait_for_space()
                pending.append(ex.submit(process_file, f, work, slice_path, class_table, None, None, drop_list_path))
            for fu in pending:
                stats.append(fu.result())
    total: dict = {"files": len(files), "class_table": ct}
    for s in stats:
        for k, v in s.items():
            if isinstance(v, int):
                total[k] = total.get(k, 0) + v
            elif k == "drops":
                d = total.setdefault("drops", {})
                for kk, vv in v.items():
                    d[kk] = d.get(kk, 0) + vv
    with open(os.path.join(work, "build_stats.json"), "w") as fh:
        json.dump({"total": total, "files": stats}, fh, indent=1, sort_keys=True)
    return total


__all__ = ["build", "process_file", "entity_strings", "choose_ala_lc", "same_name_reasons", "may_be_alive",
           "load_drop_list", "arabic_units", "style_tag"]
