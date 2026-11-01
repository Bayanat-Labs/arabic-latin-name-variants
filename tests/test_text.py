import unicodedata

from arabic_latin_name_variants.text import (arabic_token, buckwalter, clean, from_buckwalter, latin_compact, latin_fold,
                                             naive_romanise, normalize_ar, reorder_inverted, script_of, spelling_key,
                                             token_spelling)


def test_script_of():
    assert script_of("جدة") == "arabic"
    assert script_of("Jeddah") == "latin"
    assert script_of("Riyadh الرياض") == "mixed"
    assert script_of("Деревня") == "other"
    assert script_of("-") == "other"


def test_clean_drops_digits_mixed_and_parentheses():
    assert clean("Riyadh 2") == (None, "digits")
    assert clean("جدة ٢") == (None, "digits")
    assert clean("Riyadh الرياض")[0] is None
    assert clean("Mohammed Al Qahtani (footballer)") == ("Mohammed Al Qahtani", "latin")
    assert clean("محمد (لاعب)") == ("محمد", "arabic")


def test_clean_drops_addresses_and_handles():
    assert clean("@somehandle")[0] is None
    assert clean("https://example.edu/")[0] is None
    assert clean("School A@B")[0] is None


def test_clean_honorifics_only_for_people():
    assert clean("Sheikh Zayed Road")[0] == "Sheikh Zayed Road"
    assert clean("Sheikh Zayed bin Sultan", person=True)[0] == "Zayed bin Sultan"
    assert clean("الشيخ زايد", person=True)[0] == "زايد"
    assert clean("الشيخ عثمان")[0] == "الشيخ عثمان"


def test_normalize_policy():
    assert normalize_ar("إِلَى المَدْرَسَةِ") == "الي المدرسة"
    assert normalize_ar("أحمد") == "احمد"
    assert normalize_ar("جدة") == "جدة"


def test_latin_fold():
    assert latin_fold("Muḥammad ibn ʿAbd al-Wahhāb") == "muhammad ibn abd al-wahhab"
    assert latin_fold("Bahreïn") == "bahrein"
    assert latin_fold("Şeyh  Ömer") == "seyh omer"
    assert latin_fold("Straße") == "strasse"
    assert latin_compact("Al-Qahtani") == "alqahtani"


def test_buckwalter_round_trip():
    for word, bw in [("محمد", "mHmd"), ("جدة", "jdp"), ("القحطاني", "AlqHTAny"), ("مُحَمَّد", "muHam~ad"),
                     ("عبد الله", "Ebd Allh"), ("إبراهيم", "<brAhym"), ("مصطفى", "mSTfY"), ("ذئب", "*}b"),
                     ("مسؤول", "ms&wl"), ("شيء", "$y'"), ("آمنة", "|mnp"), ("ٱلله", "{llh")]:
        assert buckwalter(word) == bw
        assert unicodedata.normalize("NFC", from_buckwalter(bw)) == unicodedata.normalize("NFC", word)


def test_buckwalter_unknown_letter_is_none():
    assert buckwalter("کتاب") is None  # Persian keheh is outside the table


def test_buckwalter_round_trip_with_arabic_punctuation():
    name = "العين، إمارة أبو ظبي"
    assert buckwalter(name) == "AlEyn, <mArp >bw Zby"
    assert from_buckwalter(buckwalter(name)) == name


def test_buckwalter_skips_format_characters():
    assert buckwalter("\u202bمحمد\u200b\u202c") == "mHmd"
    assert buckwalter('محمد_"') is None  # a character outside the table


def test_arabic_token():
    assert arabic_token("دادلي،") == "دادلي"
    assert arabic_token('"بيج') == "بيج"
    assert arabic_token("روكين'") == "روكين"
    assert arabic_token("اي.") is None
    assert arabic_token("،") is None
    assert arabic_token("محمد") == "محمد"


def test_clean_drops_encoding_damage():
    assert clean("O$Brien") == (None, "encoding_damage")
    assert clean("N\u00fa\u221a\u00b1ez") == (None, "encoding_damage")


def test_naive_romanise():
    assert naive_romanise("محمد") == "mhmd"
    assert naive_romanise("القحطاني") == "alqhtani"
    assert naive_romanise("يوسف") == "yusf"


def test_reorder_inverted():
    assert reorder_inverted("Ṭabbāra, ʿAfīf") == "ʿAfīf Ṭabbāra"
    assert reorder_inverted("Jeddah") is None
    assert reorder_inverted("A, B, C") is None


def test_token_spelling():
    assert token_spelling("Thomas,") == "Thomas"
    assert token_spelling("\u200fMuḥammad\u200f") == "Muḥammad"
    assert token_spelling('"Abdullah"') == "Abdullah"
    assert token_spelling("e Sofla") == "Sofla"
    assert token_spelling("Rustai ye") == "Rustai"
    assert token_spelling("Ali-ye") == "Ali"
    assert token_spelling("Henry I") is None
    assert token_spelling("Abd Allah") == "Abd Allah"
    assert token_spelling("al-Qahtani") == "al-Qahtani"
    assert token_spelling("ʿAbd Allāh") == "ʿAbd Allāh"
    assert token_spelling("E.") == "E."
    assert token_spelling("Charles A.") is None
    assert token_spelling("A. Smith") is None
    assert token_spelling("Jones III") is None
    assert token_spelling(",") is None


def test_spelling_key():
    assert spelling_key("D.") == spelling_key("D") == "d"
    assert spelling_key("Muḥammad") == "muhammad"
    assert spelling_key("Abd al-Rahman") == "abd al-rahman"
