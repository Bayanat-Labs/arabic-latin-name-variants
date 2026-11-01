"""String handling: script detection, cleaning, Latin folding, Buckwalter and a naive romanisation.

Every function here is deterministic and depends only on the Python standard library, the
``regex`` module (Unicode script classes) and ``arabic-normalize`` (policy bayanat-norm-v1).
"""
from __future__ import annotations

import hashlib
import unicodedata
from functools import lru_cache

import regex

NORM_POLICY = "bayanat-norm-v1"

_ARABIC = regex.compile(r"\p{Script=Arabic}")
_LATIN = regex.compile(r"\p{Script=Latin}")
_LETTER = regex.compile(r"\p{L}")
_DIGIT = regex.compile(r"\p{Nd}|[⁰-⁹₀-₉²³¹]")
_OTHER_LETTER = regex.compile(r"(?=\p{L})(?!\p{Script=Arabic})(?!\p{Script=Latin})(?!\p{Script=Common})"
                              r"(?!\p{Script=Inherited}).")
_PAREN = regex.compile(r"\s*[\(\[（【][^\)\]）】]*[\)\]）】]\s*")
_SPACE = regex.compile(r"\s+")
_URL_OR_HANDLE = regex.compile(r"@|://|\bwww\.|/|\.(?:com|org|net|edu|gov|info)\b", regex.I)
# Encoding damage in Wikidata text ("O$Brien", mojibake with the square root sign).
_JUNK = regex.compile("[$\u221a]")
HARAKAT = regex.compile("[ً-ْ]")


def script_of(text: str) -> str:
    """'arabic', 'latin', 'mixed' (both), 'other' (another script, or no letters)."""
    has_ar = bool(_ARABIC.search(text))
    has_la = bool(_LATIN.search(text))
    if _OTHER_LETTER.search(text):
        return "other"
    if has_ar and has_la:
        return "mixed"
    if has_ar:
        return "arabic"
    if has_la:
        return "latin"
    return "other"


def has_digit(text: str) -> bool:
    return bool(_DIGIT.search(text))


# Honorifics and titles removed from the start of people's names (both scripts). Place and
# organisation names keep them ("Sheikh Zayed Road", "King Saud University").
AR_HONORIFICS = (
    "صاحب السمو الملكي", "صاحب السمو", "صاحبة السمو", "سمو", "الشيخ", "الشيخة", "شيخ", "السيد", "السيدة",
    "الدكتور", "الدكتورة", "د.", "الأمير", "الأميرة", "الملك", "الملكة", "الإمام", "الحاج", "الحاجة",
    "المهندس", "الأستاذ", "الأستاذة", "الفريق", "اللواء", "العقيد", "سماحة", "فضيلة",
)
LATIN_HONORIFICS = (
    "his royal highness", "her royal highness", "his highness", "her highness", "hrh", "h.r.h.", "h.h.",
    "hh", "sheikh", "shaikh", "sheik", "cheikh", "scheich", "şeyh", "sheikha", "shaikha", "cheikha",
    "shaykh", "shaik", "shaykha", "prince", "princess", "prinz", "prinzessin", "prins", "prinses", "prens",
    "príncipe", "principe", "princesa", "principessa", "king", "queen", "könig", "roi",
    "koning", "emir", "émir", "amir", "imam", "imām", "dr.", "dr", "sir", "mr.", "mrs.", "ms.", "sayyid",
    "sayed", "sayyed", "sidi", "hajj", "haji", "hadji", "al-hajj",
)
_AR_HON = sorted(AR_HONORIFICS, key=len, reverse=True)
_LA_HON = sorted(LATIN_HONORIFICS, key=len, reverse=True)


def strip_honorifics(text: str, script: str) -> str:
    """Remove one or more leading honorifics (people's names only)."""
    t = text
    changed = True
    while changed:
        changed = False
        if script == "arabic":
            for h in _AR_HON:
                if t.startswith(h + " ") and len(t) > len(h) + 1:
                    t = t[len(h) + 1:].lstrip()
                    changed = True
                    break
        else:
            low = t.casefold()
            for h in _LA_HON:
                if low.startswith(h + " ") and len(t) > len(h) + 1:
                    t = t[len(h) + 1:].lstrip()
                    changed = True
                    break
    return t


def clean(text: str, *, person: bool = False) -> tuple[str | None, str]:
    """(cleaned text or None, script). Cleaning: NFC, parenthetical disambiguators removed,
    whitespace collapsed, leading honorifics removed for people. None when the string is dropped:
    mixed script, another script, digits, a web address or handle, encoding damage ($ or the square
    root sign), or nothing left."""
    t = unicodedata.normalize("NFC", text)
    t = _PAREN.sub(" ", t)
    t = _SPACE.sub(" ", t).strip(" ,;:-–—")
    if not t:
        return None, "other"
    s = script_of(t)
    if s not in ("arabic", "latin"):
        return None, s
    if has_digit(t):
        return None, "digits"
    if _URL_OR_HANDLE.search(t):
        return None, "url_or_handle"
    if _JUNK.search(t):
        return None, "encoding_damage"
    if person:
        t = strip_honorifics(t, s)
    if not _LETTER.search(t):
        return None, "other"
    return t, s


@lru_cache(maxsize=60_000)
def normalize_ar(text: str) -> str:
    """Arabic text_norm with arabic-normalize, policy bayanat-norm-v1."""
    from arabic_normalize import normalize

    return normalize(text, NORM_POLICY)


_INVERTED = regex.compile(r"^([^,]+),\s+([^,]+)$")


def reorder_inverted(text: str) -> str | None:
    """'Surname, Given names' (a catalogue heading) -> 'Given names Surname', else None."""
    m = _INVERTED.match(text)
    if not m:
        return None
    head, tail = m.group(1).strip(), m.group(2).strip()
    if len(tail.split()) > 5 or len(head.split()) > 4:
        return None
    return f"{tail} {head}"


def is_diacritised(text: str) -> bool:
    return bool(HARAKAT.search(text))


# --------------------------------------------------------------------------- Latin fold

_APOS = dict.fromkeys(map(ord, "'’‘`´ʿʾʻʼʹ′ʽˈ"), None)
_SPECIAL = str.maketrans({"ı": "i", "ł": "l", "ø": "o", "đ": "d", "ð": "d", "ħ": "h", "ŧ": "t", "ŀ": "l",
                          "ɨ": "i", "ʃ": "sh", "ǝ": "e", "ə": "e", "ɛ": "e", "ɣ": "gh"})
_MULTI = {"ß": "ss", "æ": "ae", "œ": "oe", "þ": "th", "Æ": "ae", "Œ": "oe"}


@lru_cache(maxsize=60_000)
def latin_fold(text: str) -> str:
    """Casefolded Latin text without diacritics or apostrophe-like marks (ā→a, ḥ→h, ʿ/ʾ removed);
    hyphens and single spaces are kept."""
    t = unicodedata.normalize("NFC", text)
    for k, v in _MULTI.items():
        t = t.replace(k, v)
    t = t.casefold().translate(_SPECIAL)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.translate(_APOS)
    t = t.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    return _SPACE.sub(" ", t).strip()


def latin_compact(text: str) -> str:
    """latin_fold with spaces, hyphens and dots removed (a key for grouping spellings)."""
    return "".join(ch for ch in latin_fold(text) if ch.isalpha())


# --------------------------------------------------------------------------- spellings of a Latin word

_FORMAT_CHARS = regex.compile(r"\p{Cf}")
_TOKEN_EDGES = regex.compile(r"^[\s\"“”„‟«»,;:!?*+·&$]+|[\s\"“”„‟«»,;:!?*+·&$]+$")
_EZAFE = regex.compile(r"^(?:e|i|ye|yi)[ -](?=\p{L})", regex.I)
_EZAFE_END = regex.compile(r"(?<=\p{L})[ -](?:e|i|ye|yi)$")  # lower case only, so "Henry I" stays a regnal name
_NAME_SUFFIXES = frozenset({"jr", "sr", "i", "ii", "iii", "iv", "v", "vi"})


def token_spelling(token: str) -> str | None:
    """The spelling a Latin alignment unit gives the variant tables, or None when it holds more than
    one word's material: format characters and edge punctuation are removed, a Persian ezafe in
    front or behind ("e Sofla", "Rustai ye") is removed, and a unit with a lone initial or a
    generational suffix beside another word ("Charles A.", "Jones III") gives no spelling.
    Apostrophes and dots are kept."""
    t = _TOKEN_EDGES.sub("", _FORMAT_CHARS.sub("", token))
    t = _EZAFE.sub("", t)
    t = _EZAFE_END.sub("", t)
    words = t.split()
    if not words:
        return None
    if len(words) > 1:
        for w in words:
            bare = w.rstrip(".").casefold()
            if len(bare) == 1 or bare in _NAME_SUFFIXES:
                return None
    return t


_AR_TOKEN_EDGES = regex.compile("^[\\s\u060c,\u061b;:\"'\u00ab\u00bb|+!.\u200c]+|[\\s\u060c,\u061b;:\"'\u00ab\u00bb|+!.\u200c]+$")


def arabic_token(unit: str) -> str | None:
    """An Arabic alignment unit without edge punctuation, or None when nothing is left or it is a
    lone initial (up to two letters ending in a dot), which gives no spelling."""
    stripped = unit.strip()
    if stripped.endswith(".") and len(_LETTER.findall(stripped)) <= 2:
        return None
    t = _AR_TOKEN_EDGES.sub("", unit)
    return t if _LETTER.search(t) else None


def spelling_key(spelling: str) -> str:
    """The key that groups spellings of a word: latin_fold without a final dot ("D." and "D")."""
    return latin_fold(spelling).rstrip(".")


# --------------------------------------------------------------------------- people's name strings

# Wikidata labels and aliases are free text. On a person they sometimes hold a description or a
# statement instead of a name. human_string_reason() is the first of three checks on people's
# strings; align.same_name_words() and an optional drop list (build.load_drop_list) are the others.
# A dropped variant costs little and a false statement about a real person costs a lot, so these
# lists lean towards dropping; the whitelists hold real names read among the hits.
HUMAN_MAX_CHARS = 90  # about the longest 0.01% of Arabic and 0.02% of Latin strings on people
HUMAN_MAX_WORDS = 10  # this many words or more, with no name particle
NAME_PARTICLES = frozenset(
    "بن ابن بنت ابو ابي بني ال آل عبد ولد ibn bin ben bint binti abu abi abd al el ul de del della di da das dos du "
    "des la le van von der den ter ten zu zum zur af av ap ab ag y e of mac ould".split())

_AR_ANY = {
    "sexual_orientation": (
        "مثلي مثلية مثليين مثليات لوطي سحاقية سحاقي متحولة"
    ),
    "insult": (
        "عاهرة عاهر مومس قحبة شرموطة شرموطا شرموتة متناك متناكة منيوك معرص عرص ديوث كسمك خولات الخول حقير "
        "حقيرة حثالة تافه تافهة غبية اغبى اهبل هبلة هبيد عبيط متخلف متخلفة الزبالة قذر قذرة اقذر اوسخ وغد "
        "ملعون ملعونة كذاب كذابة الكذاب نصاب نصابة حرامي حرامية فاسدة فاشل فاشلة خاين خاينة خائن خائنة منافق "
        "منافقة الدجال معتوه شمكار نذل نذلة بلطجية مرتزق مرتزقة عنصرية سخيف سخيفة الخنزير الكلب الارنب "
        "الثعبان البراميل"
    ),
    "crime": (
        "ارهابي ارهابية ارهابيين داعشي داعشية تكفيري مجرم مجرمة مجرمين سفاح سفاحة قاتل قاتلة مغتصب مغتصبة "
        "متحرش متهم متهمة مدانة مسجون مسجونة سجين سجينة مقتول مقتولة اغتيل منتحر منتحرة جاسوس جاسوسة مختلس "
        "محتال محتالة"
    ),
    "religion": (
        "يهودية اليهودي كافرة الكافر النصراني نصراني نصرانية زنديق ملحد ملحدة مرتد مرتدة الماروني المارونية "
        "القبطي القبطية الشيعي الشيعية الديانة ديانته ديانتها معتنق"
    ),
    "ethnicity": (
        "الامازيغي الامازيغية غجرية"
    ),
    "health": (
        "مريض مريضة الجذام توفيت متوفية"
    ),
    "address": (
        "مواليد شارع"
    ),
    "political": (
        "ماسونية الماسونية اخواني اخوانية"
    ),
    "family": (
        "زوجته زوجها والدها اولاده"
    ),
    "sentence": (
        "يعتبر تعتبر يلعب تلعب يعمل تعمل الذين ويقال يقال وقيل ايضا هذه المعروف المعروفة الملقب الملقبة "
        "الشهير الشهيرة المشهورة المشتهر المشتهرة المدعو المدعوة سيرة السيرة الذاتية تعريف سابقا السابقة "
        "بالعربية المنسوب"
    ),
}

_AR_ANY_NOT_FIRST = {
    "sentence": (
        "الذي هذا وهذا يعرف ويعرف تعرف يعد تعد توفي"
    ),
    "family": (
        "والده ابوه"
    ),
}

_AR_PHRASES = {
    "description": [
        "الحزب الوطني", "امير المسلمين", "الطريقة النوربخشية", "رابطة العمل المعادية لليهود",
    ],
    "sexual_orientation": [
        "مثلي الجنس", "مثلية الجنس", "ثنائي الجنس", "ثنائية الجنس",
    ],
    "insult": [
        "ابن حرام", "ابن الحرام", "بنت حرام", "ولد حرام", "ولد الحرام", "ابن زنا", "ابن الزنا",
        "ابن الكلب", "ابن كلب", "بنت الكلب", "ابن عاهرة", "ابن شرموطة", "قليل الادب", "قليلة الادب",
        "كس ام", "مفتي البراميل",
    ],
    "ethnicity": [
        "من اصل", "من اصول", "ذو اصول", "ذات اصول", "عربي الاصل", "غير مجنسة", "يحمل الجنسية",
    ],
    "religion": [
        "عليه السلام", "عليها السلام", "عليهم السلام", "رضي الله عنه", "رضي الله عنها", "رضى الله عنه",
        "قدس سره", "صلى الله عليه", "في الاسلام", "ليس سني", "بل هو شيعي", "درزية الديانة",
    ],
    "health": [
        "رحمه الله", "رحمها الله", "يرحمه الله",
    ],
    "address": [
        "من مواليد", "من سكان", "ولد في", "ولدت في", "مكان الميلاد", "تاريخ الميلاد", "تاريخ الوفاة",
        "توفي في", "توفي ب", "توفي سنة",
    ],
    "job": [
        "رجل اعمال", "رجل الاعمال", "سيدة اعمال", "سيدة الاعمال", "رائد اعمال", "رائدة اعمال",
        "صانع محتوى", "صانعة محتوى", "حارس مرمى", "كرة القدم", "كرة قدم", "كرة السلة", "كرة اليد",
        "امين عام", "رئيس تحرير", "مجلس الامة", "مجلس النواب", "مجلس الشعب", "مجلس الشورى", "عضو مجلس",
        "يلعب لنادي", "جائزة الكرة الذهبية", "رئيس وزراء", "رئيس الوزراء", "رئيس جمهورية",
        "رئيس الجمهورية", "رئيس مجلس", "رئيس دولة", "رئيس الدولة", "رئيس نادي", "وزير سابق",
        "نائب سابق", "رئيس سابق", "لاعب سابق", "وزير الدفاع", "وزير الخارجية", "وزير الداخلية",
        "وزير المالية", "الوزير الاول",
    ],
    "crime": [
        "تاجر مخدرات", "اكل لحوم", "بارون مخدرات", "مجرم حرب", "حكم عليه", "حكم عليها", "جريمة قتل",
    ],
    "sentence": [
        "من هو", "من هي", "لا توجد", "ما هو", "هو من", "هو احد", "هي احدى", "هي من", "احد اشهر",
        "احد ابرز", "احد كبار", "احد اعلام", "احد مؤسسي", "احد رجال", "اول من", "اول رئيس", "اول وزير",
        "اول امراة", "اخر ملوك", "اخر امراء",
    ],
}

_AR_FIRST = {
    "crime": (
        "مقتل جريمة اغتيال اعدام قضية حادثة حادثه واقعة اعتقال اختفاء تفجير فضيحة عملية شنق قتل توقيف تعذيب"
    ),
    "health": (
        "وفاة انتحار"
    ),
    "political": (
        "الشهيد الشهيدة المجاهد المجاهدة المناضل المناضلة ثورة الثورة حركة جند"
    ),
    "family": (
        "زوجة قرينة مرات والد والدة اخت اخو شقيقة ابنة ابناء امراة المراة فتاة"
    ),
    "job": (
        "الفنان الفنانة الشاعر الشاعرة الكاتب الكاتبة الاعلامي الاعلامية المخرج المخرجة اللاعب اللاعبة "
        "المحامي المحامية السفير السفيرة المستشار المستشارة المفتي الممثل الممثلة المطرب المطربة الوزير "
        "الوزيرة النائب النائبة الجنرال الرسام الرسامة الصحفي الصحفية الروائي الروائية المؤلف المؤلفة المغني "
        "المغنية الموسيقار الفيلسوف المنشد المصور المصورة القارئ الراقصة المهندس المهندسة المدير المديرة "
        "الرئيس الرئيسة ممثل ممثلة كاتب كاتبة شاعر شاعرة طبيب طبيبة سياسي سياسية لاعب لاعبة فنان فنانة مغني "
        "مغنية مطرب مطربة مخرج مخرجة صحفي صحفية مهندس مهندسة محامي محامية مدرب مدربة مؤلف مؤلفة مؤسس وزيرة "
        "عضو نائب نائبة محافظ مدير مديرة خبير خبيرة مصور ملحن مخترع راقصة دكتورة اخصائي اخصائية طلبة خريجو مترجم شركة جمعية "
        "نادي مدرسة جامعة قائمة تصنيف كتاب رواية مركز متحف مستشفى الجمعية"
    ),
    "sentence": (
        "هل"
    ),
}

_AR_WHOLE = {
    "description": (
        "الخط السرطان العربية الفراعنة القائد الملكة المواقف"
    ),
    "ethnicity": (
        "جزائري جزائرية يمني يمنية مصري مصرية سوري سورية عراقي عراقية سعودي سعودية لبناني لبنانية فلسطيني "
        "فلسطينية اردني اردنية كويتي قطري اماراتي بحريني عماني ليبي تونسي مغربي مغربية سوداني موريتاني امريكي "
        "امريكية فرنسي فرنسية بريطاني بريطانية الماني ايطالي ايطالية روسي اسباني ايراني عرب عربي عربية المغرب "
        "مصر فرنسا اليمن العراق سوريا لبنان فلسطين الجزائر تونس ليبيا السودان الاردن السعودية امريكا يهودي"
    ),
    "religion": (
        "مسلم مسيحي مسيحية اليهودي كافر"
    ),
    "job": (
        "رجل الرجل امراة انسان شخص شخصية مجهول مجهولة"
    ),
    "sentence": (
        "هو هي"
    ),
}

_AR_WHITELIST = [
    "مينوهين", "ياسين", "كاتب جلبي", "ثورة حوام", "اخت حتب", "مسيلمة", "الحارث الكذاب",
    "ابو العباس السفاح", "باشا السفاح", "بن السفاح", "السفاح العباسي", "السفاح بن", "يوليان المرتد",
    "ثيودوروس الملحد", "السهروردي المقتول", "مارية القبطية", "ماريه القبطية", "ماريا القبطية",
    "شمعون القبطية", "عبد الله الشيعي", "عبدالله الشيعي", "الكاتب الاصفهاني", "ابن عمرو ابن حرام",
    "مسلم بن", "رأس الكلب", "ديميتري الدجال", "ديميترى الدجال",
]

_LA_ANY = {
    "sexual_orientation": (
        "homosexual homosexuals homosexuel homosexuelle homosexuell omosessuale homossexual homoseksual "
        "homoseksueel lesbian lesbians lesbienne lesbiana lesbica lesbisch lesbisk lezbiyen bisexual bisexuel "
        "bisexuell bisessuale biseksual biseksueel transgender transsexual transexual transsexuell "
        "transsexuelle transessuale pansexual pedophile paedophile pedofilo pedofila padophil paedophiler "
        "padophiler pedofil pedofiel maricon faggot fags sodomite sodomist tranny schwuchtel flikker frocio "
        "ricchione tortillera bollera bencong kinderschander"
    ),
    "insult": (
        "bitch bitches cunt cunts fuck fucker fucking fucked motherfucker asshole arsehole bullshit shithead "
        "dickhead dumbass nigger niggers nigga whore whores slut sluts prostitute prostitutes pornstar porn "
        "retard retarded scumbag skank hurensohn arschloch drecksack drecksau vollidiot wichser schlampe "
        "salope connard connasse encule enculer putain merde mierda hijoputa gilipollas pendejo cabron culero "
        "puttana stronzo coglione vaffanculo klootzak kankerlijer caralho cuzao orospu pezevenk "
        "serefsiz gerizekali yavsak bangsat kontol memek bajingan pelacur lonte goblok tolol knullare idiots "
        "idiot imbecile imbecil estupido estupida traitor traitors traidor traidora traitre traditore "
        "verrater verrader mentiroso mentirosa menteur menteuse lugner bugiardo leugenaar pengkhianat "
        "pembohong"
    ),
    "crime": (
        "terrorist terrorists terroriste terrorista terrorisme terorist terorista teroris murderer murderers "
        "murderess murdered killer killers rapist rapists kidnapper kidnappers trafficker traffickers "
        "smuggler gangster mobster mafioso criminal criminals convicted strangler strangled poisoner "
        "assassinated assassination beheaded perpetrator jihadist jihadists jihadi islamist extremist asesino "
        "asesina asesinos asesinado asesinada asesinato estrangulador destripador descuartizador violador "
        "narcotraficante genocida secuestrador meurtrier meurtriere meurtre tueur tueuse etrangleur eventreur "
        "violeur assassine assassinat moerder morder mordaren ermordet ermordung vergewaltiger "
        "kriegsverbrecher hingerichtet wurger massenmorder serienmorder assassino assassina assassinato "
        "assassinado strangolatore macellaio stupratore omicidio ucciso uccisa giustiziato moordenaar "
        "vermoord moord verkrachter oorlogsmisdadiger estuprador katil katili cinayeti tecavuzcu pembunuh "
        "pembunuhan pemerkosa koruptor narapidana dibunuh dieksekusi mordare valdtaktsman krigsforbrytare "
        "massacre massacred genocide vampire vampira vampier cannibal cannibale canibal kannibal kannibale maniac vergiftigster assassin assassins murder murders vampir vampiro monstruo"
    ),
    "religion": (
        "jewish jewess juif juive judio judia judisch juedisch judisk judeu joods joodse ebreo ebrea giudeo "
        "muslims moslems musulman musulmane musulmana musulmano muculmano muculmana muslimin moslim atheist "
        "atheists agnostic athee ateo atea ateu ateia ateist ateis apostate infidel kafir murtad mualaf "
        "mormon evangelical sunnite shiite chiite chiita sunnita sciita schiit schiitisch soenniet sjiiet "
        "lislam islamda islamic"
    ),
    "ethnicity": (
        "dorigine herkunft abstammung ascendencia ascendance afkomst afkomstig keturunan kokenli asilli "
        "descendiente descendente discendenza"
    ),
    "political": (
        "nazi nazis neonazi neo-nazi fascist fascista faschist communist communists commie comunista "
        "kommunist dictator dictador dictateur diktator dittatore zionist sionista zionista separatista "
        "politburo hamas hezbollah hizbullah hizbollah daesh al-qaeda alqaeda qaeda taliban guillotined"
    ),
    "health": (
        "cancer tumor tumour leukemia leucemia autistic autism schizophrenic schizophrenia dementia "
        "suicide suicides suicided suicidio suicida selbstmord zelfmoord sjalvmord intihar krankheit maladie "
        "enfermedad enfermo enferma malattia kanker kanser alcoholic addict addicted paralyzed paralysed"
    ),
    "address": (
        "strasse sokak mahallesi"
    ),
    "family": (
        "daughter daughters ehefrau weduwe gattin echtgenote mulher abuela abuelo novia grandson grandsons "
        "granddaughter boyfriend girlfriend adoptive annesi babasi kardesi selir"
    ),
    "description": (
        "orchestra orquesta orchester productions produktion fernsehproduktion gmbh inc ltd llc corporation "
        "foundation fondation stiftung fundacion institute institut university universidad academy category "
        "categoria kategorie anexo affaire discography discografia filmography bibliography wikipedia "
        "biography biografia"
    ),
}

_LA_LOWER = {
    "sentence": (
        "is was were are has had have been being who whom whose which that this these those they their them "
        "your his her him not never because but from with for by about after before when where while than "
        "then there here also like all only very est etait sont ont avec pour dans qui ist sind hatte wurde "
        "nicht fur tiene tenia fue fueron muy como para pero ahora actualmente aveva sono per heeft werd niet "
        "voor hij zij tem tinha nao com icin olan oldu idi tarafindan gibi adli adiyla adalah dengan untuk "
        "pada itu seorang telah sudah tidak inte blev and to or if in aka street origine sons nacio murio nacido nacida alzheimer"
    ),
    "job": (
        "actor actress actriz acteur actrice attore attrice schauspieler schauspielerin singer cantante "
        "chanteur chanteuse sanger sangerin zanger zangeres poet poeta poete dichter sair writer writers "
        "escritor escritora ecrivain schriftsteller scrittore scrittrice schrijver yazar author auteur autor "
        "autora autore painter pintor pintora peintre pittore pittrice maler schilder ressam footballer "
        "futbolista footballeur fussballer calciatore voetballer futebolista futbolcu player jugador jugadora "
        "joueur giocatore jogador politician politico politica politiker politicus siyasetci journalist "
        "periodista journaliste giornalista jornalista gazeteci wartawan lawyer abogado abogada avocat anwalt "
        "advocaat advogado avukat pengacara architect arquitecto architekt architetto teacher profesor "
        "professeur lehrer founder fundador fundadora fondateur grunder ceo chairman president presidente "
        "minister ministro ministre ministar member miembro former director directora directeur scientist "
        "historian historiador philosopher economist engineer ingeniero ingenieur doctor businessman "
        "businesswoman entrepreneur empresario empresaria unternehmer pengusaha rapper musician musico "
        "musicien musiker composer compositor compositeur komponist producer designer model coach ambassador "
        "diplomat athlete boxer wrestler racing penyanyi pelakon sarkici oyuncu mother father brother sister "
        "wife husband widow viuda veuve vedova witwe child children fille figlia moglie tochter dochter vrouw "
        "filha esposa esposo femme mujer"
    ),
    "description": (
        "band company companies project school club film films movie music records group team society "
        "association museum center centre award prize preis prix book novel album series show radio "
        "television history death murder killing disappearance case trial incident affair scandal list people "
        "man woman girl boy person american british indian egyptian turkish tunisian moroccan palestinian "
        "iranian mexican italian french german spanish russian arab arabic african asian european ripper "
        "schlachter gypsy dead born died unknown desconocido desconocida"
    ),
}

_LA_PHRASES = {
    "sexual_orientation": [
        "hard gay",
    ],
    "crime": [
        "the ripper", "boucher de", "der schlachter", "el carnicero", "serial killer", "mass murderer",
        "war criminal", "drug lord", "drug dealer", "murder of", "killing of", "death of",
        "disappearance of", "murdered by", "butcher of", "monster of", "beast of", "mostro di",
        "bestie von", "monstre de", "monstruo de", "carnicero de", "killer of", "the butcher",
    ],
    "insult": [
        "son of a bitch", "hijo de puta", "fils de pute", "figlio di puttana", "filho da puta",
        "pecho frio",
    ],
    "religion": [
        "was jewish", "in islam", "im islam", "en islam", "dans lislam", "dans l islam",
    ],
    "ethnicity": [
        "of origin", "de origen", "di origine", "de origem", "the gypsy", "la gitana", "el gitano",
    ],
    "health": [
        "bunuh diri", "tenia cancer", "suicide of",
    ],
    "family": [
        "son of", "daughter of", "wife of", "husband of", "mother of", "father of", "brother of",
        "sister of", "widow of", "married to", "child of", "grandson of", "granddaughter of",
        "nephew of", "niece of", "cousin of", "uncle of", "aunt of", "hijo de", "hija de", "esposa de",
        "esposo de", "viuda de", "padre de", "sobrino de", "sobrina de", "abuela de", "abuelo de",
        "novia de", "fils de", "fille de", "epouse de", "femme de", "veuve de", "frere de", "soeur de",
        "neveu de", "sohn von", "sohn des", "tochter von", "tochter des", "witwe von",
        "mutter von", "vater von", "bruder von", "schwester von", "figlio di", "figlia di",
        "moglie di", "marito di", "madre di", "padre di", "fratello di", "sorella di", "vedova di",
        "zoon van", "dochter van", "vrouw van", "moeder van", "vader van", "broer van", "zus van",
        "filho de", "filha de", "marido de", "pai de", "irmao de", "irma de", "viuva de",
        "son till", "dotter till", "hustru till", "moder till", "fader till",
    ],
    "address": [
        "born in", "born on", "born at", "living in", "lives in", "resident of", "native of",
        "natural de", "originaire de", "nacio en", "nacido en", "nacida en", "ne a", "nee a", "nato a",
        "nata a", "murio en",
    ],
    "political": [
        "leader of", "president of", "minister of", "member of", "spokesman of", "muslim brotherhood",
    ],
    "job": [
        "author of", "founder of", "ceo of", "professor of", "editor of", "director of", "creator of",
        "host of", "football player", "soccer player", "tv host", "prime minister", "editor-in-chief",
    ],
    "description": [
        "kroatisch-slowenischer bauernaufstand", "motion picture directors association", "united fruit company",
        "godfather of", "voice of", "pioneer of", "list of", "portrait of", "tomb of", "statue of",
        "the band", "known as", "also known", "a.k.a", "stage name", "pen name", "real name",
        "birth name", "nickname",
    ],
}

_LA_WHOLE = {
    "religion": (
        "muslim muslima moslem jew jews hindu buddhist sikh catholic protestant sunni shia"
    ),
    "ethnicity": (
        "egyptian palestinian moroccan tunisian turkish arab arabic mexican american"
    ),
    "description": (
        "unknown anonymous desconocido desconocida inconnu artist butcher homo image loser museum"
    ),
}

_LA_FIRST = {
    "description": (
        "actor actress actriz poet writer footballer politician journalist founder ceo band orchestra project "
        "company death murder killing suicide affaire caso"
    ),
}

_LA_WHITELIST = [
    "fuck", "spy ward", "if-jesus-christ-had-not-died", "aun porn", "nazi paikidze", "nazi-maruttash",
    "nazi-bugash", "of that ilk", "madre de dios", "nurul islam jihadi", "maolana jihadi",
    "allama jihadi", "leone ebreo", "guglielmo ebreo", "ebreo da pesaro", "takis wurger", "project itoh", "project pat",
]


def _ar_forms(word: str) -> set[str]:
    w = normalize_ar(word)
    return {w, w[:-1] + "ه"} if w.endswith("ة") else {w}


def _expand(words: dict[str, str]) -> dict[str, str]:
    return {f: kind for kind, v in words.items() for w in v.split() for f in _ar_forms(w)}


@lru_cache(maxsize=1)
def _ar_tables() -> tuple:
    anyw: dict[str, str] = {}
    for kind, v in _AR_ANY.items():
        for w in v.split():
            for f in _ar_forms(w):
                anyw[f] = kind
                if not f.startswith("ال"):
                    anyw["ال" + f] = kind
                    anyw["وال" + f] = kind
    phrases: dict[str, str] = {}
    for kind, v in _AR_PHRASES.items():
        for p in v:
            words = [normalize_ar(x) for x in p.split()]
            phrases[" ".join(words)] = kind
            if words[-1].endswith("ة"):
                phrases[" ".join(words[:-1] + [words[-1][:-1] + "ه"])] = kind
    pattern = regex.compile(r"(?<!\p{L})(?:" + "|".join(sorted(map(regex.escape, phrases), key=len, reverse=True))
                            + r")(?!\p{L})")
    return (anyw, _expand(_AR_ANY_NOT_FIRST), phrases, pattern, _expand(_AR_FIRST), _expand(_AR_WHOLE),
            tuple(" ".join(normalize_ar(x) for x in w.split()) for w in _AR_WHITELIST))


@lru_cache(maxsize=1)
def _la_tables() -> tuple:
    phrases = {p: kind for kind, v in _LA_PHRASES.items() for p in v}
    pattern = regex.compile(r"(?<!\p{L})(?:" + "|".join(sorted(map(regex.escape, phrases), key=len, reverse=True))
                            + r")(?!\p{L})")
    return ({w: k for k, v in _LA_ANY.items() for w in v.split()}, {w: k for k, v in _LA_LOWER.items() for w in v.split()},
            phrases, pattern, {w: k for k, v in _LA_WHOLE.items() for w in v.split()},
            frozenset(_LA_FIRST["description"].split()), tuple(latin_fold(w) for w in _LA_WHITELIST))


_AR_WORD = regex.compile(r"\p{Script=Arabic}+")
_LA_WORD = regex.compile(r"\p{L}+")
_SENTENCE_MARK = regex.compile("[?!\u061f;\u061b:#*=|^~<>{}]| & ")
_SUFFIX_AFTER_COMMA = regex.compile(r",\s*(?:jr|sr|ii|iii|iv|phd)\.?\s*$", regex.I)


def _structure_reason(text: str, script: str) -> str | None:
    if len(text) > HUMAN_MAX_CHARS:
        return "structure:length"
    if _SENTENCE_MARK.search(text):
        return "structure:punctuation"
    trimmed = text.rstrip(" \u060c")
    if trimmed.count(",") + trimmed.count("\u060c") >= 2 and not _SUFFIX_AFTER_COMMA.search(trimmed):
        return "structure:punctuation"
    words = text.split()
    if len(words) >= 4 and trimmed.endswith(".") and len(_LETTER.findall(words[-1])) >= 4:
        return "structure:punctuation"
    if len(words) >= HUMAN_MAX_WORDS:
        keys = {normalize_ar(w) if script == "arabic" else latin_fold(w) for w in words}
        if not keys & NAME_PARTICLES:
            return "structure:words"
    return None


def _arabic_reason(text: str) -> str | None:
    anyw, not_first, phrases, pattern, first, whole, white = _ar_tables()
    toks = _AR_WORD.findall(normalize_ar(text))
    if not toks:
        return None
    joined = " ".join(toks)
    if any(w in joined for w in white):
        return None
    if joined in whole:
        return "lexicon:" + whole[joined]
    m = pattern.search(joined)
    if m:
        return "lexicon:" + phrases[m.group(0)]
    for i, t in enumerate(toks):
        if t in anyw:
            return "lexicon:" + anyw[t]
        if i and t in not_first:
            return "lexicon:" + not_first[t]
    if toks[0] in first:
        return "lexicon:" + first[toks[0]]
    return None


def _latin_reason(text: str) -> str | None:
    anyw, lower, phrases, pattern, whole, first, white = _la_tables()
    fold = latin_fold(text)
    toks = _LA_WORD.findall(fold)
    if not toks:
        return None
    if any(w in fold for w in white):
        return None
    if fold in whole:
        return "lexicon:" + whole[fold]
    m = pattern.search(fold)
    if m:
        return "lexicon:" + phrases[m.group(0)]
    for t in toks:
        if t in anyw:
            return "lexicon:" + anyw[t]
    for raw in text.split():  # whole words written in lower case: a sign of a phrase, not of a name
        raw = raw.strip(",.;:!?()\"\u201c\u201d\u00ab\u00bb")
        if raw and raw == raw.lower() and "'" not in raw and "\u2019" not in raw:
            t = latin_fold(raw)
            if t in lower:
                return "lexicon:" + lower[t]
    if toks[0] in first:
        return "lexicon:description"
    return None


def human_string_reason(text: str, script: str) -> str | None:
    """Why a person's string is not used as a name ("lexicon:<kind>" or "structure:<kind>"), or None."""
    return _structure_reason(text, script) or (_arabic_reason(text) if script == "arabic" else _latin_reason(text))


def string_key(text: str, script: str) -> str:
    """The normalised form of a string: Arabic text_norm or latin_fold."""
    return normalize_ar(text) if script == "arabic" else latin_fold(text)


def string_sha256(text: str, script: str) -> str:
    """SHA-256 of the normalised string; the drop list holds this next to the QID, never the text."""
    return hashlib.sha256(string_key(text, script).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- Buckwalter

# Tim Buckwalter's transliteration table (one ASCII character per Arabic character), with the
# common extensions for letters outside standard Arabic (پ P, چ J, ڤ V, گ G).
BUCKWALTER = {
    "ء": "'", "آ": "|", "أ": ">", "ؤ": "&", "إ": "<", "ئ": "}",
    "ا": "A", "ب": "b", "ة": "p", "ت": "t", "ث": "v", "ج": "j",
    "ح": "H", "خ": "x", "د": "d", "ذ": "*", "ر": "r", "ز": "z",
    "س": "s", "ش": "$", "ص": "S", "ض": "D", "ط": "T", "ظ": "Z",
    "ع": "E", "غ": "g", "ـ": "_", "ف": "f", "ق": "q", "ك": "k",
    "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ى": "Y",
    "ي": "y", "ً": "F", "ٌ": "N", "ٍ": "K", "َ": "a", "ُ": "u",
    "ِ": "i", "ّ": "~", "ْ": "o", "ٰ": "`", "ٱ": "{",
    "پ": "P", "چ": "J", "ڤ": "V", "گ": "G",
    "،": ",", "؛": ";", "؟": "?",
}
_BW_PASS = set(" -.")
_SHADDA_AFTER = regex.compile("([\u064B-\u0650]+)\u0651")
BUCKWALTER_REVERSE = {v: k for k, v in BUCKWALTER.items()}


def buckwalter(text: str) -> str | None:
    """Buckwalter transliteration of an Arabic-script string, or None if a character is not in the
    table (for example Persian keheh or farsi yeh). Format characters are skipped."""
    out = []
    # NFC puts a short vowel before shadda; Buckwalter text conventionally writes shadda first.
    t = _SHADDA_AFTER.sub(lambda m: "\u0651" + m.group(1), unicodedata.normalize("NFC", text))
    for ch in t:
        if ch in BUCKWALTER:
            out.append(BUCKWALTER[ch])
        elif ch in _BW_PASS:
            out.append(ch)
        elif unicodedata.category(ch) == "Cf":
            continue
        else:
            return None
    return "".join(out)


def from_buckwalter(text: str) -> str:
    return "".join(BUCKWALTER_REVERSE.get(ch, ch) for ch in text)


# --------------------------------------------------------------------------- naive romanisation

_NAIVE = {
    "ء": "", "آ": "a", "أ": "a", "ؤ": "u", "إ": "i", "ئ": "i", "ا": "a", "ب": "b", "ة": "a", "ت": "t",
    "ث": "th", "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s",
    "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ى": "a", "ي": "y", "ٱ": "a",
    "پ": "p", "چ": "ch", "ڤ": "v", "گ": "g", "ک": "k", "ی": "y",
}


def naive_romanise(text: str) -> str:
    """A letter-by-letter romanisation of undiacritised Arabic (used by a benchmark baseline):
    long-vowel letters between consonants are read as vowels (و→u, ي→i), otherwise as w/y."""
    t = normalize_ar(text)
    words = []
    for w in t.split():
        out = []
        for i, ch in enumerate(w):
            if ch in "وي" and 0 < i < len(w):
                if i == len(w) - 1 or w[i + 1] not in "اوي":
                    out.append("u" if ch == "و" else "i")
                    continue
            out.append(_NAIVE.get(ch, ""))
        words.append("".join(out))
    return " ".join(w for w in words if w)
