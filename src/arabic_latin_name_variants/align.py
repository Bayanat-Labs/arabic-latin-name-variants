"""Token alignment between an Arabic name and a Latin name, and style tags for Latin spellings.

Arabic compounds are merged before alignment: عبد + the next word (عبد الله, عبد الرحمن), أبو/أبي/أبا +
the next word, a word + الدين or الله (نور الدين, فتح الله), and آل + the next word (آل سعود). The
article ال stays on its word. On the Latin side, words are split on spaces and hyphens, but an
article (al-, el-, ad-, ash-, ul- ...) stays on the word that follows it.

With equal unit counts the units are aligned by position. Otherwise, or when the position
alignment scores below the threshold, a dynamic programme finds the best monotone alignment that
allows one-to-one pairs, one Arabic unit to two Latin units ("عبد الله" to "Abd Allah") and two to
one, and skipped units. The alignment score is the sum of pair similarities divided by the number
of pairs plus skipped units. Units without letters (a lone dash or tatweel) are ignored.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .config import ALIGN_MIN, SAME_NAME_MIN, SAME_NAME_MIN_MERGED
from .skeleton import similarity
from .text import latin_fold, normalize_ar

_ABU = {"ابو", "ابي", "ابا"}
_TAIL = {"الدين", "الله"}
_LAT_ARTICLES = {"al", "el", "ul", "il", "ad", "ar", "as", "at", "az", "an", "ash", "adh", "ath", "ed", "er",
                 "es", "et", "ez", "en", "esh", "ud", "ur", "us", "ut", "uz", "un", "ush", "ed-", "a"}
SKIP_PENALTY = 0.3
EXTRA_PART = 0.9
_HAS_LETTER = re.compile(r"[^\W\d_]")


def arabic_units(text: str) -> list[str]:
    """Arabic name -> alignment units (compounds merged), as raw text."""
    words = [w for w in text.split() if _HAS_LETTER.search(w.replace("\u0640", ""))]
    norm = [normalize_ar(w) for w in words]
    units: list[list[str]] = []
    i = 0
    while i < len(words):
        cur = [words[i]]
        n = norm[i]
        if (n == "عبد" or n in _ABU or n == "ال" or n == "آل") and i + 1 < len(words):
            cur.append(words[i + 1])
            i += 1
        units.append(cur)
        i += 1
    merged: list[list[str]] = []
    for u in units:
        if merged and normalize_ar(u[0]) in _TAIL and len(u) == 1:
            merged[-1] = merged[-1] + u
        else:
            merged.append(u)
    return [" ".join(u) for u in merged]


def latin_units(text: str) -> list[str]:
    """Latin name -> alignment units (article kept on the next word)."""
    parts = []
    for w in (x for x in text.split() if _HAS_LETTER.search(x)):
        bits = [b for b in re.split(r"[-‐‑]", w) if b]
        k = 0
        while k < len(bits):
            b = bits[k]
            if latin_fold(b) in _LAT_ARTICLES and k + 1 < len(bits) and latin_fold(b) != "a":
                parts.append(b + "-" + bits[k + 1])
                k += 2
            else:
                parts.append(b)
                k += 1
    units: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if latin_fold(p) in ("al", "el") and i + 1 < len(parts):
            units.append(p + " " + parts[i + 1])
            i += 2
        else:
            units.append(p)
            i += 1
    return units


@lru_cache(maxsize=60_000)
def unit_sim(ar: str, la: str) -> float:
    return similarity(ar, la)


def align(ar_text: str, latin_text: str) -> tuple[list[tuple[str, str, float]], float]:
    """(pairs [(ar_unit, latin_unit, similarity)], alignment score)."""
    pairs, score, _ = align_full(ar_text, latin_text)
    return pairs, score


def align_full(ar_text: str, latin_text: str) -> tuple[list[tuple[str, str, float]], float, int]:
    """(pairs, alignment score, number of units left unmatched on either side)."""
    A = arabic_units(ar_text)
    L = latin_units(latin_text)
    if not A or not L:
        return [], 0.0, len(A) + len(L)
    best: tuple[list, float, int] | None = None
    if len(A) == len(L):
        pairs = [(a, l, unit_sim(a, l)) for a, l in zip(A, L)]
        score = sum(p[2] for p in pairs) / len(pairs)
        best = (pairs, score, 0)
        if score >= ALIGN_MIN:
            return _rounded(best)
    dp = _dp(tuple(A), tuple(L))
    if best is None or dp[1] > best[1]:
        best = dp
    return _rounded(best)


def word_coverage(ar_text: str, latin_text: str, min_sim: float = 0.8) -> bool:
    """True when some Arabic unit with two or more consonants has a Latin unit at min_sim or more."""
    from .skeleton import arabic_skeleton

    A = arabic_units(ar_text)
    L = latin_units(latin_text)
    for a in A:
        if sum(1 for _, w in arabic_skeleton(a) if w >= 1.0) < 2:
            continue
        if any(unit_sim(a, l) >= min_sim for l in L):
            return True
    return False


def _rounded(best):
    pairs, score, skips = best
    return [(a, l, round(s, 4)) for a, l, s in pairs], round(score, 4), skips


def _dp(A: tuple, L: tuple) -> tuple[list, float, int]:
    n, m = len(A), len(L)
    NEG = -1e9
    # value = sum(sim) - SKIP_PENALTY * skips; keep back-pointers
    V = [[NEG] * (m + 1) for _ in range(n + 1)]
    B: list[list] = [[None] * (m + 1) for _ in range(n + 1)]
    V[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            v = V[i][j]
            if v == NEG:
                continue
            steps = []
            if i < n and j < m:
                steps.append((1, 1, unit_sim(A[i], L[j]), (A[i], L[j])))
            if i < n and j + 1 < m:
                la = L[j] + " " + L[j + 1]
                extra = max(unit_sim(A[i], L[j]), unit_sim(A[i], L[j + 1])) >= EXTRA_PART
                steps.append((1, 2, unit_sim(A[i], la), (A[i], la, extra)))
            if i + 1 < n and j < m:
                aa = A[i] + " " + A[i + 1]
                extra = max(unit_sim(A[i], L[j]), unit_sim(A[i + 1], L[j])) >= EXTRA_PART
                steps.append((2, 1, unit_sim(aa, L[j]), (aa, L[j], extra)))
            if i < n:
                steps.append((1, 0, -SKIP_PENALTY, None))
            if j < m:
                steps.append((0, 1, -SKIP_PENALTY, None))
            for di, dj, gain, pair in steps:
                nv = v + gain - (SKIP_PENALTY if pair is not None and len(pair) > 2 and pair[2] else 0.0)
                if nv > V[i + di][j + dj]:
                    V[i + di][j + dj] = nv
                    B[i + di][j + dj] = (i, j, pair, gain)
    pairs = []
    skips = 0
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, pair, gain = B[i][j]
        if pair is None:
            skips += 1
        else:
            pairs.append((pair[0], pair[1], gain))
            if len(pair) > 2 and pair[2]:
                skips += 1  # a merge where one part alone matches: the other part is an extra word
        i, j = pi, pj
    pairs.reverse()
    denom = len(pairs) + skips
    score = sum(p[2] for p in pairs) / denom if denom else 0.0
    return pairs, score, skips


# --------------------------------------------------------------------------- same name (people)

# Words that may stand beside a name without being matched on the other side: name particles and a
# few titles. Only words of three or more letters are tested at all.
SAME_NAME_WORDS_AR = frozenset(
    "ابن بنت ابو ابي ابا بني عبد ولد سيد سيدي مولاي الحاج الشيخ السيد السيدة الامير الاميرة الملك الملكة باشا بيك خان".split())
SAME_NAME_WORDS_LA = frozenset(
    "ibn bin ben bint binti abu abi abd del dos das der den van von ter ten zum zur the las los les mac "
    "jr sr iii sir lord lady dame prince princess king queen duke duchess count countess baron baroness "
    "sheikh shaikh emir amir sultan pasha bey khan sayyid haji imam".split())
_AR_RUN = re.compile(r"[\u0621-\u063a\u0641-\u064a\u066e-\u06d3\u06d5\u06fa-\u06fc\u06ff]+")
_LA_RUN = re.compile(r"[^\W\d_]+")


@lru_cache(maxsize=200_000)
def arabic_words(text: str) -> tuple[str, ...]:
    """The normalised Arabic words of a string, tokenised here and not taken from an alignment."""
    return tuple(_AR_RUN.findall(normalize_ar(text)))


@lru_cache(maxsize=200_000)
def latin_words(text: str) -> tuple[str, ...]:
    """The folded Latin words of a string (hyphens and spaces both separate)."""
    return tuple(_LA_RUN.findall(latin_fold(text)))


def aligned_words(pairs) -> tuple[set[str], set[str]]:
    """(Arabic words, Latin words) that a row's word pairs match: a pair of single units at
    SAME_NAME_MIN or more, a merged pair (two Latin or two Arabic units) at SAME_NAME_MIN_MERGED or
    more, since a merge can carry an extra word. pairs: dicts with ar_token, latin_token and score."""
    ar: set[str] = set()
    la: set[str] = set()
    for p in pairs or ():
        a, l, s = p["ar_token"], p["latin_token"], p["score"]
        if s < SAME_NAME_MIN:
            continue
        wa, wl = arabic_words(a), latin_words(l)
        merged = len(arabic_units(a)) > 1 or len(latin_units(l)) > 1
        if merged and s < SAME_NAME_MIN_MERGED:
            continue
        ar.update(wa)
        la.update(wl)
    return ar, la


def unmatched_word(ar_text: str, latin_text: str, ar_matched: set[str], la_matched: set[str]) -> bool:
    """True when either string has a word of three or more letters that is not a listed particle or
    title and is not matched in any row of the entity (ar_matched, la_matched)."""
    for w in arabic_words(ar_text):
        if len(w) >= 3 and w not in ar_matched and w not in SAME_NAME_WORDS_AR:
            return True
    for w in latin_words(latin_text):
        if len(w) >= 3 and w not in la_matched and w not in SAME_NAME_WORDS_LA:
            return True
    return False


# --------------------------------------------------------------------------- style tags

regex_vowel = re.compile(r"[aeiouy]")
_SCHOLARLY = set("āīūḥṣḍṭẓʻʼʿʾǧḫšḏṯġẖḳẕĀĪŪḤṢḌṬẒ")
_TURKISH = set("çşğıİÇŞĞ")


def style_tag(latin: str) -> str:
    """A heuristic tag for the spelling convention of a Latin form.

    strict_letter_by_letter: letter-by-letter forms without short vowels (ISO 233 style: mḥmd,
    ẗ for ة); scholarly: diacritics of the scholarly systems (ā ḥ ʿ ...); turkish: ç ş ğ ı, or -met/-ettin
    endings; german: sch/dsch/tsch; malay_indonesian: sy for ش, -uddin, a separate Abdul; french: ou, dj, ch, é,
    è, ï; english: none of these (kh, sh, th, dh or plain letters).
    """
    if "ẗ" in latin or any(len(w) >= 3 and not regex_vowel.search(w) for w in latin_fold(latin).split()):
        return "strict_letter_by_letter"
    # á is not in the set: alone it is Spanish or Czech, and beside another scholarly letter that letter decides
    if any(c in _SCHOLARLY for c in latin):
        return "scholarly"
    f = latin_fold(latin)
    if any(c in _TURKISH for c in latin) or re.search(r"(met|ettin)\b|\bmehmed\b", f):
        return "turkish"
    if re.search(r"d?t?sch", f):
        return "german"
    if re.search(r"\bsy[aeiou]|sy[aeiou]|uddin\b|\babdul\b", f):
        return "malay_indonesian"
    if re.search(r"ou|dj|ch|[éèêëï]", latin.casefold()) or re.search(r"ou|dj", f):
        return "french"
    return "english"
