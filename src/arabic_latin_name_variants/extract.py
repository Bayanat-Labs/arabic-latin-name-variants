"""Reduce one Wikidata entity (dump JSON) to its name strings, classes and countries.

Statements are read as the Wikidata Query Service reads "truthy" statements for P31, P17 and P27
(the best rank present; deprecated values never used). Name statements (P1559, P1705, P1448,
P1813) are read at every rank except deprecated, since a former official name is still a spelling
of the name. Birth dates and descriptions are read only to decide the age filter; the record
keeps the years that filter needs, and nothing of them reaches a released file.
"""
from __future__ import annotations

import re

from .config import (ARABIC_LANGS, DESCRIPTION_LANGS, HUMAN, LATIN_LANGS, METHOD_PROP, NAME_LANGS,
                     NAME_PROPS, TRANSLIT_PROP)

RANK_ORDER = {"preferred": 2, "normal": 1}
_NAME_LANGS = frozenset(NAME_LANGS)
_AR = frozenset(ARABIC_LANGS)
_LAT = frozenset(LATIN_LANGS)

# Byte patterns for the cheap test on a raw dump line, before any JSON is parsed. The dump writes
# label and alias objects as {"language":"ar","value":...} and monolingual text values as
# {"text":...,"language":"ar"}.
_LABEL_PATS = tuple(b'{"language":"%s","value"' % lang.encode() for lang in ARABIC_LANGS)
_MONO_PATS = tuple(b'"language":"%s"}' % lang.encode() for lang in ARABIC_LANGS)


def arabic_candidate(line: bytes) -> bool:
    """True when a raw entity line may hold an ar/arz/ary label, alias or monolingual value."""
    if b'"language":"ar' not in line:
        return False
    d = line.find(b'"descriptions":')
    a = line.find(b'"aliases":', d if d >= 0 else 0)
    c = line.find(b'"claims":', a if a >= 0 else 0)
    if d < 0 or a < 0 or c < 0:
        return True  # unexpected layout: let the parser decide
    for p in _LABEL_PATS:
        if line.find(p, 0, d) >= 0 or line.find(p, a, c) >= 0:
            return True
    for p in _MONO_PATS:
        if line.find(p, c) >= 0:
            return True
    return False


def statements(ent: dict, pid: str) -> list[dict]:
    return [s for s in ent.get("claims", {}).get(pid, []) if s.get("rank") in RANK_ORDER]


def truthy(ent: dict, pid: str) -> list[dict]:
    sts = statements(ent, pid)
    if not sts:
        return []
    best = max(RANK_ORDER[s["rank"]] for s in sts)
    return [s for s in sts if RANK_ORDER[s["rank"]] == best]


def snak_value(snak: dict):
    if snak.get("snaktype") != "value":
        return None
    return snak.get("datavalue", {}).get("value")


def item_id(value) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"].startswith("Q"):
        return value["id"]
    return None


def truthy_items(ent: dict, pid: str) -> list[str]:
    out: list[str] = []
    for s in truthy(ent, pid):
        q = item_id(snak_value(s.get("mainsnak", {})))
        if q and q not in out:
            out.append(q)
    return out


def _qual_strings(st: dict, pid: str) -> list[str]:
    out = []
    for sn in (st.get("qualifiers") or {}).get(pid, []):
        v = snak_value(sn)
        if isinstance(v, str) and v not in out:
            out.append(v)
    return out


def _qual_items(st: dict, pid: str) -> list[str]:
    out = []
    for sn in (st.get("qualifiers") or {}).get(pid, []):
        q = item_id(snak_value(sn))
        if q and q not in out:
            out.append(q)
    return out


# --------------------------------------------------------------------------- age filter inputs
# The same rule as the Arab-World Wikidata slice: the latest possible birth year from P569 at any
# rank (coarse dates give the last year they can stand for), and birth-year or childhood hints in
# descriptions.


def birth_year_latest(ent: dict) -> int | None:
    years = []
    for s in ent.get("claims", {}).get("P569", []):
        v = snak_value(s.get("mainsnak", {}))
        if not isinstance(v, dict) or not isinstance(v.get("time"), str):
            continue
        t, prec = v["time"], v.get("precision", 11)
        try:
            sign = -1 if t.startswith("-") else 1
            year = sign * int(t[1:].split("-", 1)[0])
        except ValueError:
            continue
        if prec >= 9:
            years.append(year)
        elif prec == 8:
            years.append(year - year % 10 + 9 if year >= 0 else year)
        elif prec == 7:
            years.append(-(-year // 100) * 100 if year > 0 else year)
        elif prec == 6:
            years.append(-(-year // 1000) * 1000 if year > 0 else year)
    return max(years) if years else None


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_AR_MARKS = re.compile("[ً-ْٰـ]")
_YEAR = r"(1[0-9]{3}|20[0-9]{2})(?![0-9])"
_BORN = [
    re.compile(r"\b(?:born|b\.)[^0-9()]{0,20}(?:[0-9]{1,2}(?:st|nd|rd|th)?[^0-9()]{1,12})?" + _YEAR, re.I),
    re.compile(r"\bnée?\s+(?:en|le)\b[^0-9()]{0,5}(?:[0-9]{1,2}(?:er)?[^0-9()]{1,12})?" + _YEAR, re.I),
    re.compile(r"(?:مواليد|مولود|مولودة|ولد|ولدت)\s*(?:عام|سنة|في)?\s*(?:[0-9]{1,2}\s+\S+\s+)?" + _YEAR),
]
_CHILD = [
    re.compile(r"\b(?:child|baby|infant|toddler|newborn|stillborn|boy|girl|teen|teenager|teenaged|schoolboy|"
               r"schoolgirl|kid|minor|(?:son|daughter|child) of|enfant|bébé|adolescente?|garçon|fillette|"
               r"fils de|fille de)\b", re.I),
    re.compile(r"(?<![ء-ي])[وبفلك]?(?:ال)?(?:طفل|طفلة|رضيع|رضيعة|صبي|فتى|فتاة|قاصر|مراهق|مراهقة|"
               r"نجل|نجلة|ابن|ابنة|بنت)(?![ء-ي])"),
]
_SPAN = re.compile(r"\(\s*" + _YEAR + r"\s*[-–—]")
_ANY_YEAR = re.compile(r"(?<![0-9])" + _YEAR)


def description_birth_hints(texts) -> tuple[int | None, int | None]:
    """(birth year, child year) named in description texts, each the latest found, or None."""
    born, child = [], []
    for text in texts:
        t = _AR_MARKS.sub("", text.translate(_DIGITS))
        for p in _BORN:
            born += [int(m.group(1)) for m in p.finditer(t)]
        if any(p.search(t) for p in _CHILD):
            span = _SPAN.search(t)
            years = [int(span.group(1))] if span else [int(y) for y in _ANY_YEAR.findall(t)]
            if years:
                child.append(max(years))
    return (max(born) if born else None), (max(child) if child else None)


def latest_birth_year(birth_year: int | None, text_birth: int | None, text_child: int | None) -> int | None:
    known = [y for y in (birth_year, text_birth) if y is not None]
    if known:
        return max(known)
    return text_child


# --------------------------------------------------------------------------- the record


def extract(ent: dict, keep_classes=None, person_classes=frozenset({HUMAN})) -> dict | None:
    """The compact record for one entity, or None when it is out of scope.

    keep_classes: classes that reach a kept class root (None keeps every item with a P31 value).
    """
    qid = ent.get("id", "")
    if not qid.startswith("Q"):
        return None
    p31 = truthy_items(ent, "P31")
    if not p31:
        return None
    if keep_classes is not None and not any(c in keep_classes for c in p31):
        return None
    labels = ent.get("labels", {})
    aliases = ent.get("aliases", {})
    texts = []
    for lang in NAME_LANGS:
        if lang in labels:
            texts.append([lang, "label", labels[lang]["value"]])
        for a in aliases.get(lang, ()):
            texts.append([lang, "alias", a["value"]])
    mono = []
    for pid in NAME_PROPS:
        for st in statements(ent, pid):
            v = snak_value(st.get("mainsnak", {}))
            if not isinstance(v, dict) or not isinstance(v.get("text"), str):
                continue
            lang = v.get("language")
            translits = _qual_strings(st, TRANSLIT_PROP)
            if lang not in _NAME_LANGS and not translits:
                continue
            mono.append([pid, lang, v["text"], st["rank"], translits, _qual_items(st, METHOD_PROP)])
    trans = []
    for st in statements(ent, TRANSLIT_PROP):
        v = snak_value(st.get("mainsnak", {}))
        if isinstance(v, str):
            trans.append([v, _qual_items(st, METHOD_PROP), st["rank"]])

    has_ar = any(t[0] in _AR for t in texts) or any(m[1] in _AR for m in mono)
    has_lat = (any(t[0] in _LAT for t in texts) or any(m[1] in _LAT for m in mono)
               or bool(trans) or any(m[4] for m in mono))
    if not (has_ar and has_lat):
        return None
    rec = {"id": qid, "rev": ent.get("lastrevid"), "p31": p31,
           "p17": truthy_items(ent, "P17"), "p27": truthy_items(ent, "P27"),
           "t": texts, "m": mono, "x": trans}
    if HUMAN in p31 or any(c in person_classes for c in p31):
        descs = ent.get("descriptions", {})
        hints = description_birth_hints(descs[lang]["value"] for lang in DESCRIPTION_LANGS if lang in descs)
        rec["by"] = birth_year_latest(ent)
        rec["byt"], rec["cyt"] = hints
    return rec
