"""People's name strings: the lexicon and structure filter, the same-name rule, the drop list loader.
Every example here is made up."""
import hashlib

import pytest

from arabic_latin_name_variants.build import choose_ala_lc, entity_strings, load_drop_list, may_be_alive, same_name_reasons
from arabic_latin_name_variants.text import human_string_reason, string_key, string_sha256


@pytest.mark.parametrize("text", [
    "Zed Example Terrorist", "Zed Example was jewish", "Zed Example is a writer", "murdered by Example",
    "Zed Example, sons Ann, Bob", "Zed Example born in Exampletown", "daughter of Zed Example", "The Example Killer",
    "Zed Example homosexual", "Was that Zed?", "Zed, Ann, Bob", "Zed; Ann", "Label: Zed Example", "Actor Zed Example",
    "Ann Bob Carl Dan Eve Fay Gil Hal Ian Jan", "Zed " + "Example " * 15,
])
def test_latin_statements_are_dropped(text):
    assert human_string_reason(text, "latin")


@pytest.mark.parametrize("text", [
    "زيد مثلي الجنس", "مقتل زيد", "زيد ابن حرام", "زيد من مواليد شارع النيل", "الفنان زيد", "زيد من أصل يهودي",
    "زيد عليه السلام", "وفاة زيد", "زيد؟", "ممثل", "زيد، سعد، علي", "أ ب ت ث ج ح خ د ذ ر",
])
def test_arabic_statements_are_dropped(text):
    assert human_string_reason(text, "arabic")


@pytest.mark.parametrize("text,script", [
    ("Mohammed Al Qahtani", "latin"), ("Ibn Example", "latin"), ("Smith, John", "latin"), ("John Smith, Jr.", "latin"),
    ("Yehudi Example", "latin"), ("'Spy' Ward", "latin"), ("Muslim ibn Example", "latin"), ("Christian White", "latin"),
    ("Jean-Marie de la Example", "latin"), ("Ali ibn Abi Example al-Exampli ibn Zayd ibn Amr ibn Bakr", "latin"),
    ("محمد عبد الله القحطاني", "arabic"), ("رحمة الله زيد", "arabic"), ("يهودي مينوهين", "arabic"),
    ("أبو بكر بن عبد الله بن عمر بن علي بن زيد بن سعد بن خالد", "arabic"), ("شكسبير، وليام", "arabic"),
])
def test_names_are_kept(text, script):
    assert human_string_reason(text, script) is None


def _row(ar, la, rel="transliteration", pairs=()):
    return {"ar_text_raw": ar, "latin_text_raw": la, "relation": rel,
            "token_alignment": [{"ar_token": a, "latin_token": l, "score": s} for a, l, s in pairs]}


def test_same_name_rule():
    pair = [("زيد", "Zaid", 0.9), ("محمد", "Mohamed", 0.9)]
    rows = [
        _row("زيد محمد", "Zaid Mohamed", pairs=pair),                          # the same name
        _row("زيد محمد", "Example Word", "translation_exonym"),                # a translation or another name
        _row("زيد محمد", "Zaid Mohamed Terrible", "mixed", pair),              # an extra word on the Latin side
        _row("زيد محمد الطيب", "Zaid Mohamed", "mixed", pair),                 # an extra word on the Arabic side
        _row("زيد بن محمد", "Zaid ibn Mohamed", "mixed", pair),                # a listed particle, or a short word
    ]
    assert same_name_reasons(rows) == [None, "translation_exonym", "unmatched_word", "unmatched_word", None]


def test_same_name_corroboration_from_another_row():
    rows = [_row("زيد", "Zaid Mohamed", "mixed", [("زيد", "Zaid", 0.9)]),
            _row("زيد محمد", "Zaid Mohamed", pairs=[("زيد", "Zaid", 0.9), ("محمد", "Mohamed", 0.9)])]
    assert same_name_reasons(rows) == [None, None]


def test_low_or_merged_pairs_do_not_count_as_matched():
    assert same_name_reasons([_row("زيد", "Zaid", "mixed", [("زيد", "Zaid", 0.4)])]) == ["unmatched_word"]
    assert same_name_reasons([_row("زيد", "Zaid Example", "mixed", [("زيد", "Zaid Example", 0.6)])]) == ["unmatched_word"]


def test_may_be_alive():
    assert may_be_alive(None) and may_be_alive(1900) and may_be_alive(1990)
    assert not may_be_alive(1899)


def test_drop_list_removes_a_string_from_a_person_only(tmp_path):
    sha = string_sha256("Zed Example", "latin")
    path = tmp_path / "drops.yaml"
    path.write_text(f"entries:\n- {{qid: Q900000002, sha256: {sha}}}\n", encoding="utf-8")
    drops = load_drop_list(str(path))
    assert drops == {("Q900000002", sha)}
    rec = {"id": "Q900000002", "t": [["ar", "label", "زيد"], ["en", "label", "Zed Example"], ["en", "alias", "Zaid"]]}
    stats = {}
    _, la, _ = entity_strings(rec, True, stats, drops)
    assert set(la) == {"Zaid"} and stats["drop_list"] == 1
    _, la, _ = entity_strings(dict(rec, id="Q900000003"), True, {}, drops)
    assert set(la) == {"Zed Example", "Zaid"}


def test_a_dropped_string_is_never_the_ala_lc_value():
    assert choose_ala_lc("زيد", "Zayd", "Zayd", ["Zaid"]) == (None, "missing", None)
    assert choose_ala_lc("زيد", "Zayd", "Zayd", ["Zayd", "Zaid"]) == ("Zayd", "attested_p2440", "Zayd")


def test_load_drop_list(tmp_path):
    sha = "0" * 64
    path = tmp_path / "drops.yaml"
    path.write_text(f"# comment\nentries:\n- {{qid: Q1, sha256: {sha}}}\n", encoding="utf-8")
    assert load_drop_list(str(path)) == {("Q1", sha)}
    assert load_drop_list(None) == frozenset()


def test_string_key():
    assert string_key("Zéd-Exámple", "latin") == "zed-example"
    assert string_key("أحمد", "arabic") == "احمد"
    assert string_sha256("Zéd", "latin") == hashlib.sha256("zed".encode()).hexdigest()
