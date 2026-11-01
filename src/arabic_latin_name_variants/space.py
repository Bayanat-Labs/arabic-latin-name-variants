"""The data file of the static lookup page (data.js), computed from a release folder."""
from __future__ import annotations

import json
import os

import pyarrow.parquet as pq

STYLES = ["english", "french", "german", "turkish", "malay_indonesian", "scholarly", "strict_letter_by_letter"]
CHART_TOKENS = ["محمد", "جدة", "القحطاني"]
MAX_VARIANTS = 12


def space_data(release: str, stats: dict, max_bytes: int = 7_500_000, extra: dict | None = None) -> tuple[str, dict]:
    """(JavaScript text, summary). Tokens are taken by number of entities until max_bytes."""
    tv = pq.read_table(os.path.join(release, "data", "token_variants"),
                       columns=["ar_token_norm", "latin_token", "n_entities", "n_langs", "style_tag", "rank",
                                "token_entities"]).to_pylist()
    by: dict[str, list] = {}
    tok_n: dict[str, int] = {}
    for r in tv:
        by.setdefault(r["ar_token_norm"], []).append(r)
        tok_n[r["ar_token_norm"]] = r["token_entities"]
    order = sorted(by, key=lambda k: (-tok_n[k], k))
    tokens: dict[str, list] = {}
    size = 0
    for k in order:
        rows = sorted(by[k], key=lambda r: r["rank"])[:MAX_VARIANTS]
        entry = [tok_n[k], len(by[k]), [[r["latin_token"], r["n_entities"], r["n_langs"], STYLES.index(r["style_tag"])]
                                       for r in rows]]
        piece = len(json.dumps({k: entry}, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if size + piece > max_bytes:
            break
        tokens[k] = entry
        size += piece
    charts = {}
    for k in CHART_TOKENS:
        rows = sorted(by.get(k, []), key=lambda r: r["rank"])[:10]
        charts[k] = {"token_entities": tok_n.get(k, 0), "n_variants": len(by.get(k, [])),
                     "rows": [[r["latin_token"], r["n_entities"], r["n_langs"], STYLES.index(r["style_tag"])]
                              for r in rows]}
    keep = ["version", "entities", "entities_by_scope", "entities_by_class", "token_variants_rows",
            "token_variants_arabic_tokens", "clusters_rows", "clusters_with_2plus_variants", "entity_variants_rows",
            "distinct_latin_strings", "rows_by_relation"]
    data = {"stats": {k: stats[k] for k in keep}, "styles": STYLES, "tokens": tokens, "charts": charts,
            "chart_order": CHART_TOKENS,
            "tokens_in_page": len(tokens), "bench": stats.get("bench", {}).get("methods", {}),
            "bench_n": {"positive": stats.get("bench", {}).get("n_positive"),
                        "negative": stats.get("bench", {}).get("n_negative")}}
    if extra:
        data.update(extra)
    js = "window.ALNV = " + json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + ";\n"
    return js, {"tokens_in_page": len(tokens), "bytes": len(js.encode("utf-8"))}
