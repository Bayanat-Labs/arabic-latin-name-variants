import pytest

from arabic_latin_name_variants.alalc import ala_lc_style, romanise


@pytest.mark.parametrize("ar,expected", [
    ("مُحَمَّد", "Muḥammad"), ("جِدَّة", "Jiddah"), ("عَبْدُ ٱللَّٰهِ", "ʻAbd Allāh"), ("القَحْطَانِيّ", "Al-Qaḥṭānī"),
    ("أَحْمَد", "Aḥmad"), ("إِبْرَاهِيم", "Ibrāhīm"), ("مُصْطَفَى", "Muṣṭafá"), ("الرِّيَاض", "Al-Riyāḍ"),
    ("عَائِشَة", "ʻĀʼishah"), ("الشَّمْس", "Al-Shams"), ("فَاطِمَة", "Fāṭimah"), ("دُبَيّ", "Dubayy"),
    ("مُحَمَّد بْن سَلْمَان", "Muḥammad ibn Salmān"),
])
def test_romanise(ar, expected):
    assert romanise(ar) == expected


@pytest.mark.parametrize("ar", ["محمد", "مُحمد", "جدة", "عبد الله"])
def test_never_from_undiacritised(ar):
    assert romanise(ar) is None


def test_style():
    assert ala_lc_style("Muḥammad")
    assert not ala_lc_style("Mohammed")
    assert not ala_lc_style("Muḥammad Ǧamāl")  # DIN 31635 letter


@pytest.mark.parametrize("latin", ["Sánchez", "Černý", "Geislerová", "János Székely", "Wādī al Ḩasan"])
def test_style_rejects_other_systems_and_plain_diacritics(latin):
    assert not ala_lc_style(latin)


@pytest.mark.parametrize("latin", ["Yaḥyá", "ʻAbd Allāh", "Muṣṭafá ibn ʻAlī", "ʿAbd al-Raḥmān"])
def test_style_accepts_ala_lc_letters(latin):
    assert ala_lc_style(latin)
