"""Consonant skeletons and a weighted similarity between an Arabic and a Latin string.

Both strings are reduced to a sequence of consonant symbols. Arabic letters map to phonetic classes
(ب b, ت/ط t, ث th, ج j, ح/ه h, خ kh, د d, ض d, ذ dh, ز z, ظ z, ر r, س/ص s, ش sh, غ gh, ف f, ق q,
ك k, ل l, م m, ن n; ع and hamza are dropped; و, ي, ى, ة, a word-final ه and the article ال are
"soft" symbols, because they are often vowels or left out). Latin text is read with its digraphs
(kh, gh, sh, ch, th, dh, dj, dsch, sch, ph); Latin vowels weigh nothing but can stand for a soft
Arabic symbol (ي as i or e, و as u or o, ة as a or e). A symbol pair costs nothing when the Latin symbol is a usual rendering of the
Arabic one (ج can be j, g, dj or Turkish c; ق can be q, k or g). The similarity is
``1 - weighted edit distance / max(total weight)``, between 0 and 1.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

HARD = 1.0
SOFT = 0.35

# Arabic letter -> (symbol, soft?) on bayanat-norm-v1 text (alef forms already plain alef).
_AR = {
    "ب": "B", "ت": "T", "ط": "T", "ث": "TH", "ج": "J", "ح": "H", "خ": "KH", "د": "D", "ض": "DD",
    "ذ": "DH", "ز": "Z", "ظ": "ZZ", "ر": "R", "س": "S", "ص": "S", "ش": "SH", "غ": "GH", "ف": "F",
    "ق": "Q", "ك": "K", "ل": "L", "م": "M", "ن": "N", "ه": "H", "و": "W", "ي": "Y", "ة": "TM",
    "پ": "B", "چ": "SH", "ڤ": "F", "گ": "G", "ک": "K", "ی": "Y", "ؤ": "W", "ئ": "Y", "ى": "YA",
}
_AR_MARKS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_AR_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_AR_SILENT = set("اأإآٱءع")

# Arabic symbol -> Latin symbols that render it at no cost.
COMPAT = {
    "B": {"B", "P", "V"},
    "T": {"T", "TH", "D"},
    "TH": {"TH", "T", "S"},
    "J": {"J", "G", "C", "Y", "ZH"},
    "H": {"H", "CH", "KH"},
    "KH": {"KH", "H", "CH", "K"},
    "D": {"D", "DH", "T"},
    "DD": {"D", "DH", "Z"},
    "DH": {"DH", "Z", "D", "TH"},
    "Z": {"Z", "S"},
    "ZZ": {"Z", "DH", "D", "TH"},
    "R": {"R"},
    "S": {"S", "C", "Z"},
    "SH": {"SH", "CH", "S"},
    "GH": {"GH", "G", "R"},
    "F": {"F", "V"},
    "Q": {"Q", "K", "G", "C"},
    "K": {"K", "C", "Q", "CH"},
    "L": {"L"},
    "M": {"M"},
    "N": {"N"},
    "W": {"W", "V", "VU"},
    "Y": {"Y", "J", "VI"},
    "YA": {"Y", "VA", "VI"},
    "TM": {"H", "VA", "VI", "T"},
    "HF": {"H", "VA", "VI"},
    "G": {"G", "J", "K"},
}
_VOWEL_SYM = {"a": "VA", "e": "VI", "i": "VI", "o": "VU", "u": "VU", "y": "VI", "w": "VU"}

_LAT_PRE = str.maketrans({
    "ş": "ŝ", "ș": "ŝ", "š": "ŝ", "ç": "ĉ", "č": "ĉ", "ğ": "ĝ", "ž": "ẑ", "ḥ": "h", "ṣ": "s", "ḍ": "d",
    "ṭ": "t", "ẓ": "z", "ḏ": "đ", "ṯ": "ŧ", "ḫ": "x", "ẖ": "x", "ǧ": "j", "ġ": "ĝ", "ḳ": "q", "ẕ": "z",
    "ı": "i", "ł": "l", "ø": "o", "ß": "s", "æ": "a", "œ": "o", "ð": "đ", "þ": "ŧ",
})
# Placeholders kept through NFKD stripping: ŝ sh, ĉ ch, ĝ gh, ẑ zh, đ dh, ŧ th.
_KEEP = {"ŝ": "SH", "ĉ": "CH", "ĝ": "GH", "ẑ": "ZH", "đ": "DH", "ŧ": "TH"}
_DIGRAPHS = [
    ("dsch", "J"), ("dzh", "J"), ("sch", "SH"), ("tch", "CH"), ("dj", "J"), ("kh", "KH"), ("gh", "GH"),
    ("sh", "SH"), ("ch", "CH"), ("th", "TH"), ("dh", "DH"), ("zh", "ZH"), ("ph", "F"), ("ck", "K"),
]
_SINGLE = {
    "b": "B", "p": "P", "t": "T", "j": "J", "g": "G", "h": "H", "d": "D", "z": "Z", "s": "S", "f": "F",
    "v": "V", "q": "Q", "k": "K", "c": "C", "x": "KH", "l": "L", "m": "M", "n": "N", "r": "R",
}
_VOWELS = set("aeiouy")
_ARTICLES = {"al", "el", "ul", "il", "ad", "ar", "as", "at", "az", "an", "ash", "adh", "ath", "ed", "er",
             "es", "et", "ez", "en", "esh", "ud", "ur", "us", "ut", "uz", "un", "ush", "ad-", "l"}


def _fold_for_skeleton(text: str) -> str:
    t = unicodedata.normalize("NFC", text).casefold().translate(_LAT_PRE)
    t = "".join(c if c in _KEEP else "".join(x for x in unicodedata.normalize("NFKD", c)
                                             if not unicodedata.combining(x)) for c in t)
    return "".join(c for c in t if c.isalpha() or c in " -")


def _latin_words(text: str) -> list[str]:
    words = [w for w in _fold_for_skeleton(text).replace("-", " ").split() if w]
    out = []
    for i, w in enumerate(words):
        # Article forms (al-, el-, ad-, ash- ...) become "l" when another word follows.
        if i + 1 < len(words) and (w in ("al", "el", "ul", "il") or (w in _ARTICLES and len(w) > 1)):
            out.append("l")
        else:
            out.append(w)
    return out


@lru_cache(maxsize=60_000)
def latin_skeleton(text: str) -> tuple[tuple[str, float], ...]:
    """Latin string -> consonant symbols with weights."""
    seq: list[tuple[str, float]] = []
    for w in _latin_words(text):
        if w == "l":
            seq.append(("L", SOFT))
            continue
        i, n = 0, len(w)
        while i < n:
            ch = w[i]
            if ch in _KEEP:
                sym = _KEEP[ch]
                if not seq or seq[-1][0] != sym or i == 0:
                    seq.append((sym, HARD))
                i += 1
                continue
            hit = None
            for dg, sym in _DIGRAPHS:
                if w.startswith(dg, i):
                    hit = (dg, sym)
                    break
            if hit:
                seq.append((hit[1], HARD))
                i += len(hit[0])
                continue
            if ch in "yw":
                nxt = w[i + 1] if i + 1 < n else ""
                if i == 0 or (nxt and nxt in _VOWELS):
                    seq.append(("Y" if ch == "y" else "W", HARD))
                else:
                    seq.append((_VOWEL_SYM[ch], 0.0))
                i += 1
                continue
            if ch in _VOWELS:
                seq.append((_VOWEL_SYM[ch], 0.0))
                i += 1
                continue
            sym = _SINGLE.get(ch)
            if sym is None:
                i += 1
                continue
            if ch == "h" and i == n - 1 and i > 0 and w[i - 1] in _VOWELS:
                seq.append(("H", SOFT))
                i += 1
                continue
            # doubled letters (mm, dd, ll) count once
            if i > 0 and w[i - 1] == ch:
                i += 1
                continue
            seq.append((sym, HARD))
            i += 1
    return tuple(seq)


@lru_cache(maxsize=60_000)
def arabic_skeleton(text: str) -> tuple[tuple[str, float], ...]:
    """Arabic text (raw or bayanat-norm-v1) -> consonant symbols with weights."""
    t = _AR_MARKS.sub("", unicodedata.normalize("NFC", text)).translate(_AR_ALEF)
    seq: list[tuple[str, float]] = []
    for w in t.split():
        w = w.strip("-.،,")
        if not w:
            continue
        if w in ("الله", "لله"):
            seq += [("L", HARD), ("HF", SOFT)]
            continue
        if w.endswith("الله") and len(w) > 4:  # عبدالله written as one word
            head, w = w[:-4], ""
            seq += arabic_skeleton(head) + (("L", HARD), ("HF", SOFT))
            continue
        start = 0
        if w.startswith("ال") and len(w) > 2:
            seq.append(("L", SOFT))
            start = 2
        prev = ""
        n = len(w)
        for i in range(start, n):
            ch = w[i]
            if ch in _AR_SILENT:
                prev = ch
                continue
            sym = _AR.get(ch)
            if sym is None:
                prev = ch
                continue
            if ch == prev:  # a letter written twice in a row counts once
                continue
            if ch == "ه" and i == n - 1 and i > start:
                sym, weight = "HF", SOFT
            elif ch in "وي":
                weight = HARD if i == start else SOFT
            elif ch in "ةؤئى":
                weight = SOFT
            else:
                weight = HARD
            seq.append((sym, weight))
            prev = ch
    return tuple(seq)


def _distance(a: tuple, b: tuple) -> float:
    la, lb = len(a), len(b)
    prev = [0.0] * (lb + 1)
    for j in range(1, lb + 1):
        prev[j] = prev[j - 1] + b[j - 1][1]
    for i in range(1, la + 1):
        sym, wa = a[i - 1]
        comp = COMPAT.get(sym, {sym})
        cur = [prev[0] + wa] + [0.0] * lb
        for j in range(1, lb + 1):
            lsym, wl = b[j - 1]
            sub = prev[j - 1] + (0.0 if lsym in comp else (wa if wa > wl else wl))
            d = prev[j] + wa
            if d < sub:
                sub = d
            ins = cur[j - 1] + wl
            cur[j] = ins if ins < sub else sub
        prev = cur
    return prev[lb]


@lru_cache(maxsize=60_000)
def similarity_skel(a: tuple, b: tuple) -> float:
    wa = sum(w for _, w in a)
    wb = sum(w for _, w in b)
    m = max(wa, wb)
    if m == 0:
        return 0.0
    return max(0.0, 1.0 - _distance(a, b) / m)


def similarity(ar_norm: str, latin: str) -> float:
    """Consonant-skeleton similarity of an Arabic string (bayanat-norm-v1) and a Latin string."""
    return similarity_skel(arabic_skeleton(ar_norm), latin_skeleton(latin))


def skeleton_key(ar_or_latin: str, script: str) -> str:
    """The skeleton as a plain key (hard symbols only, soft ones dropped), for blocking."""
    seq = arabic_skeleton(ar_or_latin) if script == "arabic" else latin_skeleton(ar_or_latin)
    return "".join(s for s, w in seq if w == HARD)


def relation(sim: float) -> str:
    """Label from the whole-string similarity alone (the build then refines it with the word
    alignment: a transliteration that leaves a word unmatched becomes mixed, and a translation
    in which one Arabic word has a close Latin word becomes mixed)."""
    from .config import EXONYM_MAX, TRANSLIT_MIN

    if sim >= TRANSLIT_MIN:
        return "transliteration"
    if sim < EXONYM_MAX:
        return "translation_exonym"
    return "mixed"
