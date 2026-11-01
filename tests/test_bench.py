import math

from arabic_latin_name_variants.bench import (bw_fold, is_holdout, jaro_winkler, recall_at_fpr, score_bw_exact,
                                              score_jw, score_skeleton)


def test_jaro_winkler_reference_values():
    assert round(jaro_winkler("martha", "marhta"), 3) == 0.961
    assert round(jaro_winkler("dwayne", "duane"), 2) == 0.84
    assert round(jaro_winkler("dixon", "dicksonx"), 3) == 0.813
    assert jaro_winkler("", "abc") == 0.0
    assert jaro_winkler("abc", "abc") == 1.0


def test_recall_at_fpr_threshold_is_strict():
    neg = [0.1] * 98 + [0.9, 0.95]
    pos = [0.96, 0.92, 0.5]
    rec, thr = recall_at_fpr(pos, neg, 0.01)  # k = 1: threshold is the 2nd highest negative
    assert thr == 0.9 and math.isclose(rec, 2 / 3)
    rec, thr = recall_at_fpr(pos, neg, 0.001)  # k = 0: threshold is the highest negative
    assert thr == 0.95 and math.isclose(rec, 1 / 3)


def test_baseline_scores():
    assert bw_fold("محمد") == "mhmd"
    assert score_bw_exact("محمد", "Mhmd") == 1.0
    assert score_bw_exact("محمد", "Mohammed") == 0.0
    assert score_jw("محمد", "Mhmd") > score_jw("محمد", "Ali")
    assert score_skeleton("محمد", "Mohammed") == 1.0


def test_holdout_is_fixed_and_about_ten_percent():
    qs = [f"Q{i}" for i in range(1, 20001)]
    share = sum(is_holdout(q) for q in qs) / len(qs)
    assert 0.09 < share < 0.11
    assert is_holdout("Q42") == is_holdout("Q42")


def test_skeleton_score_uses_both_name_orders():
    assert score_skeleton("ستيفن ليندسي", "Lindsey, Stewen W.") >= 0.7
    assert score_skeleton("ستيفن ليندسي", "Lindsey, Stewen W.") == score_skeleton("ستيفن ليندسي", "Stewen W. Lindsey")
