"""ALA-LC romanisation fields.

ALA-LC needs the short vowels, so it is only ever:
- ``attested_p2440``: a Wikidata transliteration (P2440) whose determination method (P459) is
  ALA-LC romanization (Q603208), on the statement that holds the Arabic name;
- ``attested_variant``: a Latin label or alias of the same entity written in macron and underdot
  letters (ā ī ū ḥ ṣ ḍ ṭ ẓ ʻ ʼ ʿ ʾ) that matches the Arabic form (skeleton similarity of at least 0.8);
  it may follow IJMES, BGN/PCGN or another system and is not verified as ALA-LC;
- ``auto_from_diacritised``: derived by the rules below from an Arabic form that carries its short
  vowels (harakat). An Arabic form without enough marks gives no derived value;
- ``missing``: none of the above (the value is null).
"""
from __future__ import annotations

import re
import unicodedata

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"
SUKUN, SHADDA = "ْ", "ّ"
TANWIN = {"ً", "ٌ", "ٍ"}
DAGGER = "ٰ"
MARKS = {FATHA, DAMMA, KASRA, SUKUN, SHADDA, DAGGER} | TANWIN
_VOWEL = {FATHA: "a", DAMMA: "u", KASRA: "i"}

_CONS = {
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "ḥ", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r",
    "ز": "z", "س": "s", "ش": "sh", "ص": "ṣ", "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ", "ع": "ʻ", "غ": "gh", "ف": "f",
    "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ي": "y", "ء": "ʼ", "أ": "ʼ",
    "إ": "ʼ", "ؤ": "ʼ", "ئ": "ʼ",
}
# Latin letters of ALA-LC and close systems.
ALA_LC_LETTERS = set("āīūḥṣḍṭẓĀĪŪḤṢḌṬẒʻʼʿʾ")
_SUN = set("تثدذرزسشصضطظلن")


class Incomplete(Exception):
    pass


def _split(word: str) -> list[tuple[str, set]]:
    out: list[tuple[str, set]] = []
    for ch in word:
        if ch in MARKS:
            if not out:
                raise Incomplete
            out[-1][1].add(ch)
        elif ch == "ـ":
            continue
        else:
            out.append((ch, set()))
    return out


def _word(word: str, first: bool) -> str:
    if re.sub("[\u064B-\u0652\u0670]", "", word) in ("الله", "ٱلله", "لله"):
        return "Allāh"
    bare = re.sub("[\u064B-\u0652\u0670]", "", word)
    if bare in ("بن", "ابن", "ٱبن"):
        return "ibn"
    if bare == "بنت":
        return "bint"
    letters = _split(word)
    out = []
    i = 0
    article = False
    if len(letters) > 2 and letters[0][0] in "اٱ" and letters[1][0] == "ل" and not (
            letters[1][1] & {FATHA, DAMMA, KASRA, SHADDA}):
        out.append("al-")
        article = True
        i = 2
    n = len(letters)
    start = i
    while i < n:
        ch, m = letters[i]
        last = i == n - 1
        nxt = letters[i + 1] if i + 1 < n else None
        if ch in "اٱ":
            if i == start and not article:
                v = next((_VOWEL[x] for x in (FATHA, DAMMA, KASRA) if x in m), None)
                if v is None:
                    raise Incomplete
                out.append(v)
            else:
                out.append("ā")
            i += 1
            continue
        if ch == "آ":
            out.append("ā" if i == start else "ʼā")
            i += 1
            continue
        if ch == "ى":
            out.append("á")
            i += 1
            continue
        if ch == "ة":
            out.append("h")
            i += 1
            continue
        if ch in "أإ" and i == start and not article:
            v = "i" if ch == "إ" or KASRA in m else next((_VOWEL[x] for x in (FATHA, DAMMA) if x in m), None)
            if v is None:
                raise Incomplete
            out.append(v)
            i += 1
            if nxt and nxt[0] == "ا" and v == "a":
                out[-1] = "ā"
                i += 1
            continue
        cons = _CONS.get(ch)
        if cons is None:
            raise Incomplete
        double = SHADDA in m and not (article and i == start and ch in _SUN)
        out.append(cons + cons if double and not (i == start and not article) else cons)
        vowel = next((_VOWEL[x] for x in (FATHA, DAMMA, KASRA) if x in m), None)
        if DAGGER in m:
            out.append("ā")
            i += 1
            continue
        if vowel is None:
            if SUKUN in m or (SHADDA in m and last) or last or m & TANWIN:
                i += 1
                continue
            if nxt and nxt[0] in "اوي" and not nxt[1] & {FATHA, DAMMA, KASRA}:
                raise Incomplete  # a long vowel letter after an unmarked consonant: cannot tell
            raise Incomplete
        # long vowels and diphthongs
        if nxt is not None:
            nch, nm = nxt
            nvowel = nm & {FATHA, DAMMA, KASRA}
            if vowel == "a" and nch == "ا" and not nvowel:
                out.append("ā"); i += 2; continue
            if vowel == "a" and nch == "ى":
                out.append("á"); i += 2; continue
            if vowel == "u" and nch == "و" and not nvowel and SHADDA not in nm:
                out.append("ū"); i += 2; continue
            if vowel == "i" and nch == "ي" and not nvowel and SHADDA not in nm:
                out.append("ī"); i += 2; continue
            if vowel == "i" and nch == "ي" and SHADDA in nm:
                out.append("ī")
                i += 1
                letters[i] = ("ي", nm - {SHADDA})
                if i == n - 1:
                    i += 1
                continue
            if vowel == "u" and nch == "و" and SHADDA in nm:
                out.append("ū")
                i += 1
                letters[i] = ("و", nm - {SHADDA})
                if i == n - 1:
                    i += 1
                continue
        if last:  # pausal form: no final short vowel
            i += 1
            continue
        out.append(vowel)
        i += 1
    return _cap("".join(out))


def _cap(s: str) -> str:
    if s.startswith("al-"):
        return "al-" + _cap(s[3:])
    if s.startswith(("ʻ", "ʼ")) and len(s) > 1:
        return s[0] + s[1].upper() + s[2:]
    return s[:1].upper() + s[1:]


def romanise(text: str) -> str | None:
    """ALA-LC romanisation of a diacritised Arabic name, or None when a needed vowel is not written."""
    t = unicodedata.normalize("NFC", text).strip()
    if not re.search("[ً-ْ]", t):
        return None
    words = t.split()
    try:
        out = [_word(w, k == 0) for k, w in enumerate(words)]
    except Incomplete:
        return None
    res = " ".join(out)
    if res.startswith(("ibn ", "bint ")) or res in ("ibn", "bint"):
        res = res[:1].upper() + res[1:]
    if res.startswith("al-"):
        res = "Al-" + res[3:]
    return res


def ala_lc_style(latin: str) -> bool:
    """True when a Latin string has an ALA-LC letter and no other non-ASCII letter, apart from a
    word-final á (alef maqsura) and curly apostrophes."""
    t = unicodedata.normalize("NFC", latin)
    if not any(c in ALA_LC_LETTERS for c in t):
        return False
    for i, c in enumerate(t):
        if ord(c) < 128 or c in ALA_LC_LETTERS or c in "\u2018\u2019":
            continue
        if c == "á" and (i == len(t) - 1 or not t[i + 1].isalpha()):
            continue
        return False
    return True
