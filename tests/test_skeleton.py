import pytest

from arabic_latin_name_variants.skeleton import arabic_skeleton, latin_skeleton, relation, similarity, skeleton_key


@pytest.mark.parametrize("ar,la", [
    ("محمد", "Mohammed"), ("محمد", "Muhammad"), ("محمد", "Mehmet"), ("جدة", "Jeddah"), ("جدة", "Djeddah"),
    ("جدة", "Dschidda"), ("القحطاني", "al-Qahtani"), ("القحطاني", "Alkahtani"), ("عبد الله", "Abdullah"),
    ("عبد الله", "Abd Allah"), ("عبد الرحمن", "Abd ar-Rahman"), ("خالد", "Chalid"), ("جمال", "Cemal"),
    ("أبو ظبي", "Abu Dhabi"), ("مصطفى", "Moustapha"), ("عائشة", "Aisha"), ("دبي", "Dubai"),
])
def test_transliterations(ar, la):
    assert relation(similarity(ar, la)) == "transliteration"


@pytest.mark.parametrize("ar,la", [
    ("مصر", "Egypt"), ("الجزيرة العربية", "Arabian Peninsula"), ("البحر الأحمر", "Red Sea"),
    ("الدار البيضاء", "Casablanca"), ("ألمانيا", "Deutschland"),
])
def test_exonyms(ar, la):
    assert relation(similarity(ar, la)) == "translation_exonym"


def test_mixed_band():
    assert relation(0.5) == "mixed"
    assert relation(0.7) == "transliteration"
    assert relation(0.3999) == "translation_exonym"


def test_similarity_range_and_symmetry_of_inputs():
    s = similarity("القاهرة", "Cairo")
    assert 0.0 <= s <= 1.0
    assert similarity("", "Cairo") == 0.0
    assert similarity("محمد", "") == 0.0


def test_skeletons():
    assert [s for s, _ in arabic_skeleton("محمد")] == ["M", "H", "M", "D"]
    assert [s for s, w in latin_skeleton("Mohammed") if w > 0] == ["M", "H", "M", "D"]
    assert skeleton_key("عبد الله", "arabic") == "BDL"
    assert skeleton_key("Abdullah", "latin") == "BDL"
