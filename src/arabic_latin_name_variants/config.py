"""Fixed settings. A change to what is selected or how it is grouped is a change to this file."""
from __future__ import annotations

USER_AGENT = "BayanatLabs-OpenData/1.0 (+https://bayanatlabs.com/open; hello@bayanatlabs.com)"

DUMP_DATE = "20261026"
DUMP_URL = "https://dumps.wikimedia.org/wikidatawiki/entities/{date}/wikidata-{date}-all.json.gz"
CHECKSUM_URLS = {
    "md5": "https://dumps.wikimedia.org/wikidatawiki/entities/{date}/wikidata-{date}-md5sums.txt",
    "sha1": "https://dumps.wikimedia.org/wikidatawiki/entities/{date}/wikidata-{date}-sha1sums.txt",
}

# Name strings: labels and aliases in these languages, and monolingual name statements in them.
ARABIC_LANGS = ("ar", "arz", "ary")
LATIN_LANGS = ("en", "fr", "de", "es", "it", "nl", "pt", "tr", "id", "ms", "sv", "mul")
NAME_LANGS = ARABIC_LANGS + LATIN_LANGS
# "mul" (default for all languages) labels can be in either script; the script decides.

NAME_PROPS = {
    "P1559": "native_name",    # name in native language
    "P1705": "native_label",   # native label
    "P1448": "official_name",  # official name
    "P1813": "short_name",     # short name
}
TRANSLIT_PROP = "P2440"        # transliteration (qualifier on a name statement, or a statement)
METHOD_PROP = "P459"           # determination method (qualifier giving the transliteration system)
ALA_LC = "Q603208"
DIN_31635 = "Q258311"
BUCKWALTER = "Q4983561"
METHODS = {ALA_LC: "ALA-LC", DIN_31635: "DIN 31635", BUCKWALTER: "Buckwalter"}

# Class groups (same rule as the Arab-World Wikidata slice): for each P31 class, count the P279
# steps up to each group's roots; the nearest root decides, ties go to the earlier group.
CLASS_GROUPS: list[tuple[str, dict[str, str]]] = [
    ("human", {"Q5": "human"}),
    ("place", {
        "Q2221906": "geographic location",
        "Q56061": "administrative territorial entity",
        "Q618123": "geographical feature",
        "Q486972": "human settlement",
        "Q811979": "architectural structure",
    }),
    ("organisation", {"Q43229": "organization"}),
    ("work", {"Q386724": "work"}),
]
KEPT_GROUPS = ("human", "place", "organisation")
HUMAN = "Q5"

# Wikimedia-internal and bibliographic classes (with all subclasses) are removed.
EXCLUDED_CLASSES: dict[str, str] = {
    "Q4167836": "Wikimedia category",
    "Q11266439": "Wikimedia template",
    "Q4167410": "Wikimedia disambiguation page",
    "Q13406463": "Wikimedia list article",
    "Q15184295": "Wikimedia module",
    "Q14204246": "Wikimedia project page",
    "Q13442814": "scholarly article",
    "Q17442446": "Wikimedia internal item",
}

# People whose latest possible birth year is this many years or fewer before the dump year are
# removed. Birth dates and descriptions are read for this filter only and are never released.
MIN_AGE_YEARS = 18
DESCRIPTION_LANGS = ("ar", "arz", "ary", "en", "fr", "mul")

# Properties that are never released (as a value or a qualifier). The release holds names, class
# and countries only, so none of these can appear; validate.py checks the column set.
BLOCKED_PROPS = {
    "P140": "religion or worldview", "P172": "ethnic group", "P91": "sexual orientation",
    "P1050": "medical condition", "P102": "member of political party", "P569": "date of birth",
    "P570": "date of death", "P22": "father", "P25": "mother", "P26": "spouse", "P40": "child",
    "P3373": "sibling", "P1038": "relative", "P451": "unmarried partner", "P8810": "parent",
    "P3448": "stepparent", "P1290": "godparent", "P53": "family", "P1971": "number of children",
}

# People who may be alive: a name pair is kept only when the two strings read as the same name.
# Latest possible birth year this year or later, or none known, counts as may be alive.
SAME_NAME_BIRTH_YEAR = 1900
# A word counts as matched in a word pair at this similarity, or the second value for a merged pair.
SAME_NAME_MIN = 0.5
SAME_NAME_MIN_MERGED = 0.7

# Relation filter thresholds on the consonant-skeleton similarity.
TRANSLIT_MIN = 0.7
EXONYM_MAX = 0.4
ALIGN_MIN = 0.65
# A word pair inside a kept alignment counts as a variant only from this similarity.
PAIR_MIN = 0.7
# A token variant is listed when it is seen on at least this many entities.
TOKEN_MIN_ENTITIES = 2
