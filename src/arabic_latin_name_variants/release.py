"""Write the release folder (Parquet configs) and compute its statistics from the released files."""
from __future__ import annotations

import glob
import json
import os

import pyarrow.parquet as pq

from . import __version__
from .aggregate import _con, clusters, sql_path, token_occurrences, token_variants, write_entity_variants
from .bench import Lookup, build_pairs, evaluate, is_holdout, write_pairs


def write_release(work: str, out: str, temp: str | None = None, consume: bool = False) -> dict:
    con = _con(temp=temp)
    data = os.path.join(out, "data")
    ev_dir = os.path.join(data, "entity_variants")
    ev = write_entity_variants(work, ev_dir, consume=consume)
    ev_glob = os.path.join(ev_dir, "*.parquet")
    token_occurrences(ev_dir, work)
    tv = token_variants(work, con=con)
    os.makedirs(os.path.join(data, "token_variants"), exist_ok=True)
    pq.write_table(tv, os.path.join(data, "token_variants", "train-00000-of-00001.parquet"), compression="zstd",
                   row_group_size=100_000)
    os.makedirs(os.path.join(data, "clusters"), exist_ok=True)
    n_clusters = clusters(ev_glob, con=con, out_path=os.path.join(data, "clusters", "train-00000-of-00001.parquet"))
    pairs = build_pairs(sql_path(ev_glob), con=con)
    write_pairs(pairs, os.path.join(data, "name_matching_bench"))
    held = {q for q in con.execute(
        f"SELECT DISTINCT qid FROM read_parquet('{sql_path(os.path.join(work, 'tokens', '*.parquet'))}')").fetchall()
        for q in q if is_holdout(q)}
    train_tv = token_variants(work, exclude_qids=held, con=con)
    res = evaluate(pairs, Lookup(train_tv))
    res["lookup_table_rows"] = train_tv.num_rows
    res["held_out_entities_with_tokens"] = len(held)
    os.makedirs(os.path.join(out, "meta"), exist_ok=True)
    with open(os.path.join(out, "meta", "bench_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    return {"entity_variants": ev, "token_variants": tv.num_rows, "clusters": n_clusters,
            "bench_pairs": pairs.num_rows, "bench": res}


def compute_stats(out: str) -> dict:
    """Every number the card states, computed from the released files."""
    con = _con()
    d = os.path.join(out, "data")
    ev = sql_path(os.path.join(d, "entity_variants", "*.parquet"))
    tv = sql_path(os.path.join(d, "token_variants", "*.parquet"))
    cl = sql_path(os.path.join(d, "clusters", "*.parquet"))
    bp = sql_path(os.path.join(d, "name_matching_bench", "*.parquet"))
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    st: dict = {"version": __version__}
    st["entity_variants_rows"] = q(f"SELECT count(*) FROM read_parquet('{ev}')")[0][0]
    st["entities"] = q(f"SELECT count(DISTINCT qid) FROM read_parquet('{ev}')")[0][0]
    st["entities_by_scope"] = {k: v for k, v in q(
        f"SELECT scope, count(DISTINCT qid) FROM read_parquet('{ev}') GROUP BY 1 ORDER BY 1")}
    st["entities_by_scope_class"] = {}
    for scope, cls, n in q(f"""SELECT scope, entity_class, count(DISTINCT qid) FROM read_parquet('{ev}')
                               GROUP BY 1, 2 ORDER BY 1, 2"""):
        st["entities_by_scope_class"].setdefault(scope, {})[cls] = n
    st["entities_by_class"] = {k: v for k, v in q(
        f"SELECT entity_class, count(DISTINCT qid) FROM read_parquet('{ev}') GROUP BY 1 ORDER BY 1")}
    st["rows_by_relation"] = {k: v for k, v in q(
        f"SELECT relation, count(*) FROM read_parquet('{ev}') GROUP BY 1 ORDER BY 1")}
    st["distinct_arabic_strings"] = q(f"SELECT count(DISTINCT ar_text_raw) FROM read_parquet('{ev}')")[0][0]
    st["distinct_latin_strings"] = q(f"SELECT count(DISTINCT latin_text_raw) FROM read_parquet('{ev}')")[0][0]
    st["latin_transliteration_strings"] = q(
        f"SELECT count(DISTINCT latin_text_raw) FROM read_parquet('{ev}') WHERE relation = 'transliteration'")[0][0]
    st["latin_rows_by_lang"] = {k: v for k, v in q(f"""
        SELECT l, count(*) FROM (SELECT unnest(latin_langs) AS l FROM read_parquet('{ev}'))
        GROUP BY 1 ORDER BY 2 DESC, 1""")}
    st["arabic_strings_by_lang"] = {k: v for k, v in q(f"""
        SELECT l, count(*) FROM (SELECT DISTINCT qid, ar_text_raw, unnest(ar_langs) AS l FROM read_parquet('{ev}'))
        GROUP BY 1 ORDER BY 2 DESC, 1""")}
    st["ala_lc_origin_arabic_strings"] = {k: v for k, v in q(f"""
        SELECT ala_lc_origin, count(*) FROM (SELECT DISTINCT qid, ar_text_raw, ala_lc_origin FROM read_parquet('{ev}'))
        GROUP BY 1 ORDER BY 1""")}
    st["din_31635_arabic_strings"] = q(f"""SELECT count(*) FROM (SELECT DISTINCT qid, ar_text_raw
        FROM read_parquet('{ev}') WHERE din_31635 IS NOT NULL)""")[0][0]
    st["din_31635_method_qid_items"] = q(f"""SELECT count(DISTINCT qid) FROM read_parquet('{ev}')
        WHERE list_contains(method_qids, 'Q258311')""")[0][0]
    st["p2440_latin_rows"] = q(f"""SELECT count(*) FROM read_parquet('{ev}')
        WHERE list_contains(latin_kinds, 'p2440_transliteration')""")[0][0]
    st["buckwalter_null_arabic_strings"] = q(f"""SELECT count(*) FROM (SELECT DISTINCT ar_text_raw
        FROM read_parquet('{ev}') WHERE buckwalter IS NULL)""")[0][0]
    st["aligned_rows"] = q(f"SELECT count(*) FROM read_parquet('{ev}') WHERE alignment_score >= 0.65")[0][0]
    top = q(f"""SELECT c, count(DISTINCT qid) FROM (SELECT qid, unnest(countries) AS c FROM read_parquet('{ev}')
               WHERE scope = 'arab_world') GROUP BY 1 ORDER BY 2 DESC, 1""")
    st["arab_world_entities_by_country"] = {k: v for k, v in top}
    st["token_variants_rows"] = q(f"SELECT count(*) FROM read_parquet('{tv}')")[0][0]
    st["token_variants_arabic_tokens"] = q(f"SELECT count(DISTINCT ar_token_norm) FROM read_parquet('{tv}')")[0][0]
    st["token_variants_by_style"] = {k: v for k, v in q(
        f"SELECT style_tag, count(*) FROM read_parquet('{tv}') GROUP BY 1 ORDER BY 1")}
    st["clusters_rows"] = q(f"SELECT count(*) FROM read_parquet('{cl}')")[0][0]
    st["clusters_with_2plus_variants"] = q(f"SELECT count(*) FROM read_parquet('{cl}') WHERE n_variants >= 2")[0][0]
    st["cluster_variants_total"] = q(f"SELECT sum(n_variants) FROM read_parquet('{cl}')")[0][0]
    st["bench_pairs"] = {k: v for k, v in q(
        f"SELECT pair_type, count(*) FROM read_parquet('{bp}') GROUP BY 1 ORDER BY 1")}
    st["bench_pairs_by_scope"] = {k: v for k, v in q(
        f"SELECT scope, count(*) FROM read_parquet('{bp}') GROUP BY 1 ORDER BY 1")}
    st["bench_hard_negatives_top_skeleton_score"] = q(
        f"SELECT count(*) FROM read_parquet('{bp}') WHERE label = 0 AND skeleton_sim >= 1")[0][0]
    examples = {}
    for tok in ("محمد", "جدة", "القحطاني", "عبد الله"):
        examples[tok] = [
            {"latin": r[0], "n_entities": r[1], "share": float(r[2])} for r in q(f"""
            SELECT latin_token, n_entities, share FROM read_parquet('{tv}') WHERE ar_token_norm = '{tok}'
            ORDER BY rank LIMIT 12""")]
        examples[tok + "#variants"] = q(
            f"SELECT count(*) FROM read_parquet('{tv}') WHERE ar_token_norm = '{tok}'")[0][0]
    st["examples"] = examples
    bstats = os.path.join(out, "meta", "build_stats.json")
    if os.path.exists(bstats):
        with open(bstats, encoding="utf-8") as fh:
            human_filter = json.load(fh).get("human_string_filter")
        if human_filter:
            st["human_string_filter"] = human_filter
    bres = os.path.join(out, "meta", "bench_results.json")
    if os.path.exists(bres):
        with open(bres, encoding="utf-8") as fh:
            st["bench"] = json.load(fh)
    files = sorted(glob.glob(os.path.join(d, "*", "*.parquet")))
    st["files"] = {os.path.relpath(f, out).replace(os.sep, "/"): os.path.getsize(f) for f in files}
    return st
