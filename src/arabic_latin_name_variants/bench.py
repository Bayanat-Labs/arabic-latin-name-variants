"""name_matching_bench: a rule-built Arabic-Latin name matching test set and four baselines.

Split: 10% of entities are held out by QID (a fixed hash of the QID). Positives pair an Arabic name
of a held-out entity with a Latin name of the same entity that the relation filter calls a
transliteration. Hard negatives pair an Arabic name of one held-out entity with a Latin name of a
different held-out entity when the two have a consonant-skeleton similarity of at least 0.5, the
Arabic names differ, and the Latin name is not attested for any entity with the same Arabic name.

Baselines (higher score = more likely the same name):
(a) bw_fold_exact: the Buckwalter form folded to plain letters equals the folded Latin name;
(b) jaro_winkler_naive: Jaro-Winkler similarity of a naive letter-by-letter romanisation and the
    folded Latin name;
(c) skeleton: the consonant-skeleton similarity;
(d) cluster_lookup: each Arabic unit scores 1 when the aligned Latin unit is a listed variant in a
    token_variants table built without the held-out entities, else 0.8 x its skeleton similarity;
    the name score is the alignment score over units.
Recall is reported at a false positive rate of 1% and 0.1%: the threshold is the (k+1)-th highest
negative score with k = floor(rate x negatives), and a pair counts as matched only above it.
Recall of the positives of each scope is also given at the 1% threshold of all negatives.
"""
from __future__ import annotations

import hashlib
import math
import os
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .align import arabic_units, latin_units
from .skeleton import arabic_skeleton, latin_skeleton, similarity
from .text import buckwalter, latin_compact, naive_romanise, normalize_ar, reorder_inverted, spelling_key, token_spelling

HOLDOUT_SALT = "arabic-latin-name-variants/holdout/v1:"
N_POS = 5000
N_NEG = 5000
NEG_MIN_SIM = 0.5
METHODS = ["bw_fold_exact", "jaro_winkler_naive", "skeleton", "cluster_lookup"]
BENCH_SCHEMA = pa.schema([
    ("pair_id", pa.string()), ("label", pa.int8()), ("pair_type", pa.string()), ("ar_name", pa.string()),
    ("latin_name", pa.string()), ("qid_ar", pa.string()), ("qid_latin", pa.string()),
    ("entity_class", pa.string()), ("scope", pa.string()), ("skeleton_sim", pa.float32()),
])


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def is_holdout(qid: str, mod: int = 10) -> bool:
    return _h(HOLDOUT_SALT + qid) % mod == 0


# --------------------------------------------------------------------------- similarity functions


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    if s1 == s2:
        return 1.0 if s1 else 0.0
    l1, l2 = len(s1), len(s2)
    if not l1 or not l2:
        return 0.0
    rng = max(l1, l2) // 2 - 1
    m1 = [False] * l1
    m2 = [False] * l2
    matches = 0
    for i, c in enumerate(s1):
        lo, hi = max(0, i - rng), min(l2, i + rng + 1)
        for j in range(lo, hi):
            if not m2[j] and s2[j] == c:
                m1[i] = m2[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    t = 0
    k = 0
    for i in range(l1):
        if m1[i]:
            while not m2[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
    t //= 2
    jaro = (matches / l1 + matches / l2 + (matches - t) / matches) / 3
    prefix = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a != b:
            break
        prefix += 1
    return jaro + prefix * p * (1 - jaro)


_BW_FOLD = str.maketrans({"A": "a", "E": None, "H": "h", "S": "s", "D": "d", "T": "t", "Z": "z", "x": "kh",
                          "$": "sh", "v": "th", "*": "dh", "g": "gh", "p": "a", "Y": "a", "'": None, ">": None,
                          "<": None, "|": "a", "&": None, "}": None, "{": "a", "`": None, "~": None, "o": None,
                          "a": None, "u": None, "i": None, "F": None, "N": None, "K": None, "_": None,
                          " ": None, "-": None, ".": None})


def bw_fold(ar: str) -> str:
    bw = buckwalter(ar) or ""
    return bw.translate(_BW_FOLD).lower()


def score_bw_exact(ar: str, la: str) -> float:
    return 1.0 if bw_fold(ar) == latin_compact(la) else 0.0


def score_jw(ar: str, la: str) -> float:
    return jaro_winkler(naive_romanise(ar).replace(" ", ""), latin_compact(la))


def score_skeleton(ar: str, la: str) -> float:
    """Consonant-skeleton similarity; a catalogue form "Surname, Given" is compared in both orders."""
    return max(similarity(ar, la), similarity(ar, reorder_inverted(la) or la))


class Lookup:
    def __init__(self, table: pa.Table):
        self.variants: dict[str, set[str]] = defaultdict(set)
        for ar, fold in zip(table.column("ar_token_norm").to_pylist(), table.column("latin_fold").to_pylist()):
            self.variants[ar].add(fold)

    def unit(self, a: str, l: str) -> float:
        spelled = token_spelling(l)
        if spelled and spelling_key(spelled) in self.variants.get(normalize_ar(a), ()):
            return 1.0
        return 0.8 * similarity(a, l)

    def score(self, ar: str, la: str) -> float:
        A, L = arabic_units(ar), latin_units(la)
        if not A or not L:
            return 0.0
        if len(A) == len(L):
            pos = sum(self.unit(a, l) for a, l in zip(A, L)) / len(A)
        else:
            pos = 0.0
        return max(pos, _dp_score(A, L, self.unit))


def _dp_score(A, L, unit, skip: float = 0.3) -> float:
    n, m = len(A), len(L)
    NEG = -1e9
    V = [[(NEG, 0, 0.0)] * (m + 1) for _ in range(n + 1)]  # (objective, units counted, sum)
    V[0][0] = (0.0, 0, 0.0)
    for i in range(n + 1):
        for j in range(m + 1):
            obj, cnt, tot = V[i][j]
            if obj == NEG:
                continue
            steps = []
            if i < n and j < m:
                steps.append((1, 1, unit(A[i], L[j])))
            if i < n and j + 1 < m:
                steps.append((1, 2, unit(A[i], L[j] + " " + L[j + 1])))
            if i + 1 < n and j < m:
                steps.append((2, 1, unit(A[i] + " " + A[i + 1], L[j])))
            for di, dj, g in steps:
                cand = (obj + g, cnt + 1, tot + g)
                if cand[0] > V[i + di][j + dj][0]:
                    V[i + di][j + dj] = cand
            if i < n:
                cand = (obj - skip, cnt + 1, tot)
                if cand[0] > V[i + 1][j][0]:
                    V[i + 1][j] = cand
            if j < m:
                cand = (obj - skip, cnt + 1, tot)
                if cand[0] > V[i][j + 1][0]:
                    V[i][j + 1] = cand
    _, cnt, tot = V[n][m]
    return tot / cnt if cnt else 0.0


# --------------------------------------------------------------------------- building the pairs


def build_pairs(variants_glob: str, con=None) -> pa.Table:
    """5,000 positive and 5,000 hard-negative pairs from held-out entities."""
    import duckdb

    con = con or duckdb.connect()
    qids = [r[0] for r in con.execute(f"SELECT DISTINCT qid FROM read_parquet('{variants_glob}')").fetchall()]
    held_qids = pa.table({"qid": pa.array(sorted(q for q in qids if is_holdout(q)), type=pa.string())})
    del qids
    con.register("held_qids_arrow", held_qids)
    con.execute("CREATE OR REPLACE TEMP TABLE held_q AS SELECT qid FROM held_qids_arrow")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE held_rows AS
        SELECT qid, scope, entity_class, ar_text_raw, ar_text_norm, latin_text_raw, latin_fold
        FROM read_parquet('{variants_glob}') WHERE relation = 'transliteration' AND qid IN (SELECT qid FROM held_q)""")
    held = con.execute("""SELECT * FROM held_rows
        ORDER BY CAST(substr(qid, 2) AS BIGINT), ar_text_raw, latin_text_raw""").fetchall()
    # Latin folds attested (relation other than translation/exonym) for each Arabic name of a held-out row
    attested: dict[str, set[str]] = defaultdict(set)
    for arn, folds in con.execute(f"""
            SELECT ar_text_norm, list(DISTINCT latin_fold) FROM read_parquet('{variants_glob}')
            WHERE relation != 'translation_exonym' AND ar_text_norm IN (SELECT DISTINCT ar_text_norm FROM held_rows)
            GROUP BY 1""").fetchall():
        attested[arn] = set(folds)
    # positives: at most two pairs per entity, picked by hash order
    per_entity: dict[str, int] = defaultdict(int)
    pos = []
    for r in sorted(held, key=lambda r: _h("pos:" + r[0] + "|" + r[3] + "|" + r[5])):
        if per_entity[r[0]] >= 2:
            continue
        per_entity[r[0]] += 1
        pos.append(r)
        if len(pos) >= N_POS:
            break
    # hard negatives: block Latin names on their first two hard skeleton symbols
    latin_pool: dict[str, list] = defaultdict(list)
    seen_latin = set()
    for q, scope, cls, ar, arn, la, lf in held:
        if (q, lf) in seen_latin:
            continue
        seen_latin.add((q, lf))
        key = _block(latin_skeleton(la))
        if key:
            latin_pool[key].append((q, arn, la, lf))
    for k in latin_pool:
        latin_pool[k].sort(key=lambda x: _h("negpool:" + x[0] + "|" + x[2]))
    neg = []
    used_ar, used_la = set(), set()
    arabic_side = {}
    for q, scope, cls, ar, arn, la, lf in held:
        arabic_side.setdefault((q, arn), (q, scope, cls, ar, arn))
    for q, scope, cls, ar, arn in sorted(arabic_side.values(), key=lambda r: _h("neg:" + r[0] + "|" + r[3])):
        if (q, arn) in used_ar:
            continue
        key = _block(arabic_skeleton(ar))
        cands = latin_pool.get(key, [])
        start = _h("start:" + q + "|" + ar) % len(cands) if cands else 0
        for off in range(min(len(cands), 60)):
            q2, arn2, la2, lf2 = cands[(start + off) % len(cands)]
            if q2 == q or arn2 == arn or lf2 in attested.get(arn, ()) or (q2, lf2) in used_la:
                continue
            s = score_skeleton(ar, la2)
            if s >= NEG_MIN_SIM:
                neg.append((q, scope, cls, ar, la2, q2, s))
                used_ar.add((q, arn))
                used_la.add((q2, lf2))
                break
        if len(neg) >= N_NEG:
            break
    out = []
    for q, scope, cls, ar, arn, la, lf in pos:
        out.append({"label": 1, "pair_type": "positive", "ar_name": ar, "latin_name": la, "qid_ar": q,
                    "qid_latin": q, "entity_class": cls, "scope": scope, "skeleton_sim": round(score_skeleton(ar, la), 4)})
    for q, scope, cls, ar, la2, q2, s in neg:
        out.append({"label": 0, "pair_type": "hard_negative", "ar_name": ar, "latin_name": la2, "qid_ar": q,
                    "qid_latin": q2, "entity_class": cls, "scope": scope, "skeleton_sim": round(s, 4)})
    out.sort(key=lambda r: _h("order:" + r["qid_ar"] + "|" + r["ar_name"] + "|" + r["latin_name"]))
    for i, r in enumerate(out):
        r["pair_id"] = f"p{i:05d}"
    return pa.Table.from_pylist(out, schema=BENCH_SCHEMA)


def _block(seq) -> str:
    hard = [s for s, w in seq if w >= 1.0]
    return "".join(hard[:2]) if len(hard) >= 2 else ""


# --------------------------------------------------------------------------- evaluation


def recall_at_fpr(pos: list[float], neg: list[float], fpr: float) -> tuple[float, float]:
    """(recall, threshold): pairs count as matched when their score is above the threshold."""
    ns = sorted(neg, reverse=True)
    k = math.floor(fpr * len(ns) + 1e-9)
    thr = ns[k] if k < len(ns) else -math.inf
    rec = sum(1 for s in pos if s > thr) / len(pos) if pos else 0.0
    return rec, thr


def _finite(x: float):
    return None if x in (math.inf, -math.inf) else x


def evaluate(pairs: pa.Table, lookup: Lookup) -> dict:
    fns = {"bw_fold_exact": score_bw_exact, "jaro_winkler_naive": score_jw, "skeleton": score_skeleton,
           "cluster_lookup": lookup.score}
    ar = pairs.column("ar_name").to_pylist()
    la = pairs.column("latin_name").to_pylist()
    lab = pairs.column("label").to_pylist()
    scopes = pairs.column("scope").to_pylist()
    res = {"n_positive": sum(lab), "n_negative": len(lab) - sum(lab), "methods": {},
           "recall_at_fpr_1pct_by_scope": {}}
    for m in METHODS:
        sc = [round(fns[m](a, b), 6) for a, b in zip(ar, la)]
        pos = [s for s, y in zip(sc, lab) if y == 1]
        neg = [s for s, y in zip(sc, lab) if y == 0]
        r1, t1 = recall_at_fpr(pos, neg, 0.01)
        r01, t01 = recall_at_fpr(pos, neg, 0.001)
        res["methods"][m] = {"recall_at_fpr_1pct": round(r1, 4), "threshold_1pct": _finite(t1),
                             "recall_at_fpr_0_1pct": round(r01, 4), "threshold_0_1pct": _finite(t01)}
        for scope in sorted(set(scopes)):
            sp = [s for s, y, c in zip(sc, lab, scopes) if y == 1 and c == scope]
            res["recall_at_fpr_1pct_by_scope"].setdefault(scope, {"n_positive": len(sp)})[m] = (
                round(sum(1 for s in sp if s > t1) / len(sp), 4) if sp else None)
    return res


def write_pairs(tbl: pa.Table, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "test.parquet")
    pq.write_table(tbl, path, compression="zstd")
    return path
