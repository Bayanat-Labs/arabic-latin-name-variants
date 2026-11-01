from arabic_latin_name_variants.align import align, arabic_units, latin_units, style_tag


def test_arabic_compounds():
    assert arabic_units("محمد عبد الله القحطاني") == ["محمد", "عبد الله", "القحطاني"]
    assert arabic_units("أبو بكر") == ["أبو بكر"]
    assert arabic_units("نور الدين محمود") == ["نور الدين", "محمود"]
    assert arabic_units("فتح الله") == ["فتح الله"]
    assert arabic_units("آل سعود") == ["آل سعود"]
    assert arabic_units("محمد بن سلمان") == ["محمد", "بن", "سلمان"]


def test_latin_units():
    assert latin_units("Mohammed Al Qahtani") == ["Mohammed", "Al Qahtani"]
    assert latin_units("Abd ar-Rahman") == ["Abd", "ar-Rahman"]
    assert latin_units("Nur ud-Din") == ["Nur", "ud-Din"]
    assert latin_units("Jean-Pierre") == ["Jean", "Pierre"]


def test_align_equal_counts():
    pairs, score = align("محمد عبد الله القحطاني", "Mohammed Abdullah Al-Qahtani")
    assert [(a, b) for a, b, _ in pairs] == [("محمد", "Mohammed"), ("عبد الله", "Abdullah"),
                                              ("القحطاني", "Al-Qahtani")]
    assert score >= 0.9


def test_align_one_to_two():
    pairs, score = align("عبد الله", "Abd Allah")
    assert pairs[0][:2] == ("عبد الله", "Abd Allah")
    assert score >= 0.9
    pairs, score = align("محمد عبد العزيز", "Muhammad Abdul Aziz")
    assert ("عبد العزيز", "Abdul Aziz") in [(a, b) for a, b, _ in pairs]


def test_align_rejects_translation():
    _, score = align("جامعة الملك سعود", "King Saud University")
    assert score < 0.65


def test_style_tags():
    assert style_tag("Muḥammad") == "scholarly"
    assert style_tag("mḥmd") == "strict_letter_by_letter"
    assert style_tag("Mehmet") == "turkish"
    assert style_tag("Dschidda") == "german"
    assert style_tag("Djeddah") == "french"
    assert style_tag("Syarif") == "malay_indonesian"
    assert style_tag("Jeddah") == "english"


def test_extra_word_is_counted():
    from arabic_latin_name_variants.align import align_full

    pairs, score, skips = align_full("سالم ناصر", "Salem Ali Nasser")
    assert skips == 1
    assert [(a, b) for a, b, _ in pairs] == [("سالم", "Salem"), ("ناصر", "Nasser")]
    assert align_full("عبد الله", "Abd Allah")[2] == 0
    assert align_full("أبو بكر", "Abu Bakr")[2] == 0


def test_word_coverage():
    from arabic_latin_name_variants.align import word_coverage

    assert word_coverage("البحرين", "Kingdom of Bahrain")
    assert not word_coverage("مصر", "Egypt")
    assert not word_coverage("البحر الأحمر", "Red Sea")


def test_lone_marks_are_not_units():
    assert arabic_units("سوس ـ ماسة") == ["سوس", "ماسة"]
    assert latin_units("Souss - Massa") == ["Souss", "Massa"]


def test_style_tag_acute_a_is_not_scholarly():
    assert style_tag("Fernández") != "scholarly"
    assert style_tag("Abdalá") != "scholarly"
    assert style_tag("Muṣṭafá") == "scholarly"
