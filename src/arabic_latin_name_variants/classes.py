"""Class groups and excluded classes from the subclass (P279) table taken from the same dump."""
from __future__ import annotations

import glob
import io
import json
import os
from collections import defaultdict

from .config import CLASS_GROUPS, EXCLUDED_CLASSES, HUMAN


def read_jsonl_zst(path: str):
    import zstandard

    with open(path, "rb") as fh:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in reader:
            if line.strip():
                yield json.loads(line)


def load_parents(folder: str) -> dict[str, list[str]]:
    """class QID -> P279 parents, from the classes_*.jsonl.zst files written by the stream."""
    parents: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(folder, "classes_*.jsonl.zst"))):
        for row in read_jsonl_zst(path):
            parents[row["id"]] = row["p279"]
    return parents


def children_map(parents: dict[str, list[str]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for c, ps in parents.items():
        for p in ps:
            children[p].append(c)
    return children


def distances(children: dict[str, list[str]], roots) -> dict[str, int]:
    """Each class that reaches one of the roots through P279, with the fewest steps needed."""
    dist = {r: 0 for r in roots}
    frontier = list(roots)
    while frontier:
        nxt = []
        for c in frontier:
            d = dist[c] + 1
            for ch in children.get(c, ()):
                if ch not in dist:
                    dist[ch] = d
                    nxt.append(ch)
        frontier = nxt
    return dist


def descendants(children: dict[str, list[str]], roots) -> set[str]:
    seen = set(roots)
    stack = list(roots)
    while stack:
        for ch in children.get(stack.pop(), ()):
            if ch not in seen:
                seen.add(ch)
                stack.append(ch)
    return seen


class Classifier:
    """Assign a class group to a list of P31 classes, and flag excluded classes."""

    def __init__(self, parents: dict[str, list[str]]):
        children = children_map(parents)
        self.group_dist = [(g, distances(children, roots)) for g, roots in CLASS_GROUPS]
        self.excluded = set().union(*(descendants(children, [r]) for r in EXCLUDED_CLASSES))
        self.person_classes = descendants(children, [HUMAN])
        self.n_classes = len(parents)

    def is_person(self, p31) -> bool:
        return any(c in self.person_classes for c in p31)

    def group(self, p31) -> str:
        """The group whose root is the fewest P279 steps from one of the classes; ties by group order."""
        best = None
        for i, (g, dist) in enumerate(self.group_dist):
            for c in p31:
                d = dist.get(c)
                if d is not None and (best is None or (d, i) < best[:2]):
                    best = (d, i, g)
        return best[2] if best else "other"

    def is_excluded(self, p31) -> bool:
        return any(c in self.excluded for c in p31)

    def keep_classes(self, groups) -> set[str]:
        """Every class that reaches the roots of the given groups (for the stream's filter)."""
        out: set[str] = set(self.person_classes)
        for g, dist in self.group_dist:
            if g in groups:
                out |= set(dist)
        return out


def compact_table(clf: Classifier) -> dict:
    """The parts of the class table the build needs, small enough to load in each worker."""
    dist: dict[str, list] = {}
    for gi, (_g, d) in enumerate(clf.group_dist):
        for c, v in d.items():
            dist.setdefault(c, [None] * len(clf.group_dist))[gi] = v
    return {"groups": [g for g, _ in clf.group_dist], "dist": dist,
            "excluded": sorted(clf.excluded), "person": sorted(clf.person_classes), "n_classes": clf.n_classes}


class CompactClassifier:
    """Same answers as Classifier, from compact_table() output. Each class keeps only its nearest
    (distance, group order) pair, which is all the nearest-root rule needs."""

    def __init__(self, table: dict):
        self.groups = table["groups"]
        self.best = {}
        for c, ds in table["dist"].items():
            self.best[c] = min((d, i) for i, d in enumerate(ds) if d is not None)
        self.excluded = frozenset(table["excluded"])
        self.person_classes = frozenset(table["person"])
        self.n_classes = table["n_classes"]

    def is_person(self, p31) -> bool:
        return any(c in self.person_classes for c in p31)

    def is_excluded(self, p31) -> bool:
        return any(c in self.excluded for c in p31)

    def group(self, p31) -> str:
        hits = [self.best[c] for c in p31 if c in self.best]
        return self.groups[min(hits)[1]] if hits else "other"
