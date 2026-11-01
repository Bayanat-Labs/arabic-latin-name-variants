"""Release tables from the build parts: entity_variants, token_variants and clusters."""
from __future__ import annotations

import hashlib
import os

import pyarrow as pa
import pyarrow.parquet as pq

from .align import style_tag
from .config import ALIGN_MIN, PAIR_MIN, TOKEN_MIN_ENTITIES
from .text import arabic_token, buckwalter, normalize_ar, spelling_key, token_spelling

S = pa.string()
F32 = pa.float32()
I32 = pa.int32()
LS = pa.list_(S)

TOKEN_VARIANTS_SCHEMA = pa.schema([
    ("ar_token_norm", S), ("ar_token_raw", S), ("ar_buckwalter", S), ("latin_token", S), ("latin_fold", S),
    ("forms", LS), ("n_entities", I32), ("n_entities_arab_world", I32), ("n_entities_global", I32),
    ("n_langs", I32), ("langs", LS), ("entity_classes", LS), ("style_tag", S), ("rank", I32),
    ("share", F32), ("token_entities", I32), ("mean_score", F32),
])
VARIANT_T = pa.struct([("latin", S), ("latin_fold", S), ("n_entities", I32), ("n_langs", I32), ("langs", LS),
                       ("style_tag", S)])
CLUSTERS_SCHEMA = pa.schema([
    ("cluster_id", S), ("ar_text_norm", S), ("ar_text_raw", S), ("buckwalter", S), ("canonical_latin", S),
    ("variants", pa.list_(VARIANT_T)), ("n_variants", I32), ("n_entities", I32), ("entity_classes", LS),
    ("scopes", LS), ("qids", LS),
])


def _con(memory_limit: str = "3GB", threads: int = 4, temp: str | None = None):
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=true")
    if temp:
        con.execute(f"SET temp_directory='{sql_path(temp)}'")
    return con


def sql_path(path: str) -> str:
    """A file path or glob for DuckDB SQL: forward slashes (Windows too), quotes doubled."""
    return path.replace("\\", "/").replace("'", "''")


def to_arrow(res) -> pa.Table:
    """An Arrow table from a DuckDB result (the method name differs across DuckDB versions)."""
    for name in ("to_arrow_table", "fetch_arrow_table", "arrow"):
        fn = getattr(res, name, None)
        if fn is not None:
            t = fn()
            return t.read_all() if isinstance(t, pa.RecordBatchReader) else t
    raise RuntimeError("DuckDB result has no Arrow export")


def qid_key(q: str) -> int:
    return int(q[1:])


def write_entity_variants(work: str, out_dir: str, rows_per_file: int = 2_000_000, consume: bool = False) -> dict:
    """Gather the variant parts into the release files, one file set per scope. With consume, each part
    is deleted once copied (the parts and the release never both hold everything)."""
    import glob

    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for scope in ("arab_world", "global"):
        parts = sorted(glob.glob(os.path.join(work, "variants", scope, "*.parquet")))
        n = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
        counts[scope] = n
        nfiles = max(1, -(-n // rows_per_file))
        k, in_file, writer = 0, 0, None
        for p in parts:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=100_000):
                tbl = pa.Table.from_batches([batch])
                while tbl.num_rows:
                    if writer is None:
                        path = os.path.join(out_dir, f"{scope}-{k:05d}-of-{nfiles:05d}.parquet")
                        writer = pq.ParquetWriter(path, tbl.schema, compression="zstd", compression_level=9)
                    take = min(tbl.num_rows, rows_per_file - in_file)
                    writer.write_table(tbl.slice(0, take), row_group_size=100_000)
                    in_file += take
                    tbl = tbl.slice(take)
                    if in_file >= rows_per_file:
                        writer.close()
                        writer, in_file, k = None, 0, k + 1
            del pf
            if consume:
                os.remove(p)
        if writer is not None:
            writer.close()
        elif n == 0:
            from .build import VARIANT_SCHEMA

            pq.write_table(VARIANT_SCHEMA.empty_table(), os.path.join(out_dir, f"{scope}-00000-of-00001.parquet"))
    return counts


TOKEN_SCHEMA = pa.schema([
    ("qid", S), ("scope", S), ("entity_class", S), ("ar_token_norm", S), ("ar_token_raw", S),
    ("latin_token", S), ("latin_token_fold", S), ("latin_langs", LS), ("score", F32),
])
_FLOAT32_EPS = 1e-6  # scores are stored as float32, so 0.7 reads back slightly below 0.7


def token_occurrences(ev_dir: str, work: str) -> int:
    """Token occurrence parts (work/tokens) from the entity_variants files: the word pairs of alignments
    that score at least ALIGN_MIN, with a pair similarity of at least PAIR_MIN and a clean spelling."""
    import glob
    from functools import lru_cache

    out_dir = os.path.join(work, "tokens")
    os.makedirs(out_dir, exist_ok=True)
    spell = lru_cache(maxsize=500_000)(token_spelling)
    key = lru_cache(maxsize=500_000)(spelling_key)
    norm = lru_cache(maxsize=500_000)(normalize_ar)
    ar_unit = lru_cache(maxsize=500_000)(arabic_token)
    cols = ["qid", "scope", "entity_class", "latin_langs", "token_alignment", "alignment_score"]
    total = 0
    for path in sorted(glob.glob(os.path.join(ev_dir, "*.parquet"))):
        writer = pq.ParquetWriter(os.path.join(out_dir, os.path.basename(path)), TOKEN_SCHEMA, compression="zstd")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=cols):
            rows = []
            for r in batch.to_pylist():
                if r["alignment_score"] is None or r["alignment_score"] < ALIGN_MIN - _FLOAT32_EPS:
                    continue
                for p in r["token_alignment"]:
                    if p["score"] < PAIR_MIN - _FLOAT32_EPS:
                        continue
                    y = spell(p["latin_token"])
                    a = ar_unit(p["ar_token"])
                    if y is None or a is None:
                        continue
                    rows.append({"qid": r["qid"], "scope": r["scope"], "entity_class": r["entity_class"],
                                 "ar_token_norm": norm(a), "ar_token_raw": a,
                                 "latin_token": y, "latin_token_fold": key(y), "latin_langs": r["latin_langs"],
                                 "score": p["score"]})
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=TOKEN_SCHEMA))
                total += len(rows)
        writer.close()
    return total


def token_variants(work: str, exclude_qids: set[str] | None = None, con=None,
                   min_entities: int = TOKEN_MIN_ENTITIES) -> pa.Table:
    """token_variants from the token occurrence parts (optionally without some entities)."""
    con = con or _con()
    src = sql_path(os.path.join(work, "tokens", "*.parquet"))
    con.execute(f"CREATE OR REPLACE TEMP VIEW occ_all AS SELECT * FROM read_parquet('{src}')")
    if exclude_qids:
        con.execute("CREATE OR REPLACE TEMP TABLE excl(qid VARCHAR)")
        con.executemany("INSERT INTO excl VALUES (?)", [(q,) for q in sorted(exclude_qids)])
        con.execute("CREATE OR REPLACE TEMP VIEW occ AS SELECT * FROM occ_all WHERE qid NOT IN (SELECT qid FROM excl)")
    else:
        con.execute("CREATE OR REPLACE TEMP VIEW occ AS SELECT * FROM occ_all")
    rows = con.execute(f"""
        WITH forms AS (
            SELECT ar_token_norm, latin_token_fold, latin_token, count(*) AS c
            FROM occ GROUP BY ALL),
        arforms AS (
            SELECT ar_token_norm, ar_token_raw, count(*) AS c FROM occ GROUP BY ALL),
        langs AS (
            SELECT ar_token_norm, latin_token_fold, list(DISTINCT l ORDER BY l) AS langs
            FROM (SELECT ar_token_norm, latin_token_fold, unnest(latin_langs) AS l FROM occ)
            GROUP BY ALL),
        agg AS (
            SELECT ar_token_norm, latin_token_fold,
                   count(DISTINCT qid) AS n_entities,
                   count(DISTINCT CASE WHEN scope = 'arab_world' THEN qid END) AS n_aw,
                   count(DISTINCT CASE WHEN scope = 'global' THEN qid END) AS n_gl,
                   list(DISTINCT entity_class ORDER BY entity_class) AS classes,
                   round(avg(score), 4) AS mean_score
            FROM occ GROUP BY ALL),
        tok AS (
            SELECT ar_token_norm, count(DISTINCT qid) AS token_entities FROM occ GROUP BY ALL)
        SELECT a.ar_token_norm,
               (SELECT list(ar_token_raw ORDER BY c DESC, ar_token_raw)[1] FROM arforms f
                 WHERE f.ar_token_norm = a.ar_token_norm) AS ar_token_raw,
               a.latin_token_fold,
               (SELECT list(latin_token ORDER BY c DESC, latin_token)[1:10] FROM forms f
                 WHERE f.ar_token_norm = a.ar_token_norm AND f.latin_token_fold = a.latin_token_fold) AS forms,
               a.n_entities, a.n_aw, a.n_gl, coalesce(l.langs, []) AS langs, a.classes, a.mean_score,
               t.token_entities
        FROM agg a JOIN tok t USING (ar_token_norm)
        LEFT JOIN langs l USING (ar_token_norm, latin_token_fold)
        WHERE a.n_entities >= {int(min_entities)}
        ORDER BY a.ar_token_norm, a.n_entities DESC, len(coalesce(l.langs, [])) DESC,
                 list_contains(coalesce(l.langs, []), 'en') DESC, a.latin_token_fold
    """).fetchall()
    out = {f.name: [] for f in TOKEN_VARIANTS_SCHEMA}
    rank, prev = 0, None
    for (ar, ar_raw, fold, forms, n, n_aw, n_gl, langs, classes, mean_score, tok_n) in rows:
        rank = rank + 1 if ar == prev else 1
        prev = ar
        out["ar_token_norm"].append(ar)
        out["ar_token_raw"].append(ar_raw)
        out["ar_buckwalter"].append(buckwalter(ar))
        out["latin_token"].append(forms[0])
        out["latin_fold"].append(fold)
        out["forms"].append(list(forms))
        out["n_entities"].append(n)
        out["n_entities_arab_world"].append(n_aw)
        out["n_entities_global"].append(n_gl)
        out["n_langs"].append(len(langs))
        out["langs"].append(list(langs))
        out["entity_classes"].append(list(classes))
        out["style_tag"].append(style_tag(forms[0]))
        out["rank"].append(rank)
        out["share"].append(round(n / tok_n, 4) if tok_n else None)
        out["token_entities"].append(tok_n)
        out["mean_score"].append(float(mean_score))
    return pa.table(out, schema=TOKEN_VARIANTS_SCHEMA)


def cluster_id(ar_text_norm: str) -> str:
    return "c" + hashlib.sha1(ar_text_norm.encode("utf-8")).hexdigest()[:12]


def clusters(variants_glob: str, con=None, out_path: str | None = None, buckets: int = 16):
    """Full-name clusters on identical Arabic text_norm across entities (transliteration rows).

    Names are processed in buckets (the first hex digit of the MD5 of text_norm) so that memory stays
    small; with out_path the rows are written bucket by bucket and the row count is returned,
    otherwise the table is returned."""
    con = con or _con()
    src = sql_path(variants_glob)
    writer = None
    tables = []
    total = 0
    digits = "0123456789abcdef"
    step = max(1, 16 // buckets)
    for k in range(0, 16, step):
        keys = ", ".join(f"'{d}'" for d in digits[k:k + step])
        rows = con.execute(f"""
            WITH m AS (
                SELECT qid, scope, entity_class, ar_text_norm, ar_text_raw, latin_fold, latin_text_raw, latin_langs
                FROM read_parquet('{src}')
                WHERE relation = 'transliteration' AND substr(md5(ar_text_norm), 1, 1) IN ({keys})),
            forms AS (SELECT ar_text_norm, latin_fold, latin_text_raw, count(*) AS c FROM m GROUP BY ALL),
            arforms AS (SELECT ar_text_norm, ar_text_raw, count(*) AS c FROM m GROUP BY ALL),
            best_ar AS (SELECT ar_text_norm, list(ar_text_raw ORDER BY c DESC, ar_text_raw)[1] AS ar_raw
                        FROM arforms GROUP BY ALL),
            langs AS (
                SELECT ar_text_norm, latin_fold, list(DISTINCT l ORDER BY l) AS langs
                FROM (SELECT ar_text_norm, latin_fold, unnest(latin_langs) AS l FROM m) GROUP BY ALL),
            v AS (SELECT ar_text_norm, latin_fold, count(DISTINCT qid) AS n FROM m GROUP BY ALL),
            vv AS (
                SELECT v.ar_text_norm, v.latin_fold, v.n, coalesce(l.langs, []) AS langs,
                       (SELECT list(latin_text_raw ORDER BY c DESC, latin_text_raw)[1] FROM forms f
                         WHERE f.ar_text_norm = v.ar_text_norm AND f.latin_fold = v.latin_fold) AS latin
                FROM v LEFT JOIN langs l USING (ar_text_norm, latin_fold)),
            vagg AS (
                SELECT ar_text_norm,
                       list({{'latin': latin, 'latin_fold': latin_fold, 'n': n, 'langs': langs}}
                            ORDER BY n DESC, len(langs) DESC, list_contains(langs, 'en') DESC, latin_fold) AS variants
                FROM vv GROUP BY ALL),
            ent AS (
                SELECT ar_text_norm, count(DISTINCT qid) AS n_entities,
                       list(DISTINCT entity_class ORDER BY entity_class) AS classes,
                       list(DISTINCT scope ORDER BY scope) AS scopes,
                       list(DISTINCT qid ORDER BY CAST(substr(qid, 2) AS BIGINT)) AS qids
                FROM m GROUP BY ALL)
            SELECT e.ar_text_norm, b.ar_raw, a.variants, e.n_entities, e.classes, e.scopes, e.qids
            FROM ent e JOIN best_ar b USING (ar_text_norm) JOIN vagg a USING (ar_text_norm)
            ORDER BY e.ar_text_norm
        """).fetchall()
        out = {f.name: [] for f in CLUSTERS_SCHEMA}
        for ar, ar_raw, variants, n_ent, classes, scopes, qids in rows:
            vs = [{"latin": x["latin"], "latin_fold": x["latin_fold"], "n_entities": x["n"],
                   "n_langs": len(x["langs"]), "langs": list(x["langs"]), "style_tag": style_tag(x["latin"])}
                  for x in variants]
            out["cluster_id"].append(cluster_id(ar))
            out["ar_text_norm"].append(ar)
            out["ar_text_raw"].append(ar_raw)
            out["buckwalter"].append(buckwalter(ar_raw))
            out["canonical_latin"].append(vs[0]["latin"] if vs else None)
            out["variants"].append(vs)
            out["n_variants"].append(len(vs))
            out["n_entities"].append(n_ent)
            out["entity_classes"].append(list(classes))
            out["scopes"].append(list(scopes))
            out["qids"].append(list(qids))
        del rows
        tbl = pa.table(out, schema=CLUSTERS_SCHEMA)
        total += tbl.num_rows
        if out_path:
            if writer is None:
                writer = pq.ParquetWriter(out_path, CLUSTERS_SCHEMA, compression="zstd", compression_level=9)
            writer.write_table(tbl, row_group_size=100_000)
        else:
            tables.append(tbl)
    if out_path:
        if writer is None:
            pq.write_table(CLUSTERS_SCHEMA.empty_table(), out_path)
        else:
            writer.close()
        return total
    return pa.concat_tables(tables) if tables else CLUSTERS_SCHEMA.empty_table()
