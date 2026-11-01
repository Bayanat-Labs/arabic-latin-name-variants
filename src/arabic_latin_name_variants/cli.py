"""Command line: arabic-latin-name-variants <command> ..."""
from __future__ import annotations

import argparse
import json
import os
import sys


def _class_filter(args) -> int:
    from .classes import Classifier, load_parents
    from .config import KEPT_GROUPS

    clf = Classifier(load_parents(args.stream_dir))
    keep = sorted(clf.keep_classes(KEPT_GROUPS), key=lambda q: int(q[1:]))
    person = sorted(clf.person_classes, key=lambda q: int(q[1:]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "keep_classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("".join(q + "\n" for q in keep))
    with open(os.path.join(args.out, "person_classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("".join(q + "\n" for q in person))
    print(json.dumps({"keep_classes": len(keep), "person_classes": len(person)}))
    return 0


def _build(args) -> int:
    from .build import build

    total = build(args.stream_dir, args.work, args.slice, workers=args.workers, drop_list_path=args.drop_list)
    print(json.dumps(total, indent=1, sort_keys=True))
    return 0


def _release(args) -> int:
    from .release import write_release

    res = write_release(args.work, args.out, temp=args.temp)
    print(json.dumps(res, indent=1, sort_keys=True, default=str))
    return 0


def _stats(args) -> int:
    from .release import compute_stats

    st = compute_stats(args.out)
    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)
    with open(os.path.join(args.out, "meta", "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, sort_keys=True, ensure_ascii=False)
    print(json.dumps({k: st[k] for k in ("entities", "entity_variants_rows", "token_variants_rows",
                                         "clusters_rows")}, indent=1))
    return 0


def _lookup(args) -> int:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from .text import normalize_ar

    tv = pq.read_table(os.path.join(args.release, "data", "token_variants"))
    key = normalize_ar(args.arabic)
    rows = tv.filter(pc.equal(tv.column("ar_token_norm"), key)).to_pylist()
    for r in sorted(rows, key=lambda r: r["rank"])[: args.limit]:
        print(f"{r['rank']:3d}  {r['latin_token']:30s} {r['n_entities']:7d} entities  {r['n_langs']:2d} langs  "
              f"{r['style_tag']}")
    return 0 if rows else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "stream":
        from .stream import main as stream_main

        return stream_main(argv[1:])
    ap = argparse.ArgumentParser(prog="arabic-latin-name-variants",
                                 description="Arabic-Latin name variants from the Wikidata JSON dump.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stream", help="stream the dump once (see `stream --help`)")
    p = sub.add_parser("class-filter", help="keep and person class lists from a stream's class table")
    p.add_argument("--stream-dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_class_filter)
    p = sub.add_parser("build", help="name records -> entity_variants and token parts")
    p.add_argument("--stream-dir", required=True)
    p.add_argument("--slice", required=True, help="Arab-World Wikidata slice entities Parquet (file or folder)")
    p.add_argument("--work", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--drop-list", help="optional drop list of people's strings: one {qid: Q1, sha256: ...} entry per line")
    p.set_defaults(fn=_build)
    p = sub.add_parser("release", help="build parts -> release Parquet configs and benchmark")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--temp", help="DuckDB spill folder")
    p.set_defaults(fn=_release)
    p = sub.add_parser("stats", help="compute meta/stats.json from a release folder")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_stats)
    p = sub.add_parser("lookup", help="Latin spellings of an Arabic name token")
    p.add_argument("arabic")
    p.add_argument("--release", required=True)
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(fn=_lookup)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
