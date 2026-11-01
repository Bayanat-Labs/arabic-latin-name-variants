"""Benchmark chart (PNG and SVG) with Matplotlib (optional dependency: pip install ".[chart]")."""
from __future__ import annotations

LABELS = {
    "bw_fold_exact": "Buckwalter, exact match",
    "jaro_winkler_naive": "Jaro-Winkler, naive romanisation",
    "skeleton": "Consonant skeleton",
    "cluster_lookup": "Spelling-table lookup",
}
ORDER = ["bw_fold_exact", "jaro_winkler_naive", "skeleton", "cluster_lookup"]
S0, S1 = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0d", "#52514e", "#e1e0d9", "#fcfcfb"


def bench_chart(results: dict, out_stem: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = results["methods"]
    names = [LABELS[k] for k in ORDER if k in m]
    r1 = [m[k]["recall_at_fpr_1pct"] for k in ORDER if k in m]
    r01 = [m[k]["recall_at_fpr_0_1pct"] for k in ORDER if k in m]
    fig, ax = plt.subplots(figsize=(8, 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    h = 0.34
    ys = list(range(len(names)))[::-1]
    for y, a, b in zip(ys, r1, r01):
        ax.barh(y + h / 2 + 0.02, a, height=h, color=S0, zorder=3)
        ax.barh(y - h / 2 - 0.02, b, height=h, color=S1, zorder=3)
        ax.text(a + 0.01, y + h / 2 + 0.02, f"{a:.1%}", va="center", fontsize=9, color=MUTED)
        ax.text(b + 0.01, y - h / 2 - 0.02, f"{b:.1%}", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    n = f"{results['n_positive']:,} matching and {results['n_negative']:,} hard non-matching Arabic-Latin pairs"
    fig.text(0.01, 0.97, "Recall of name-matching baselines", fontsize=12, color=INK, ha="left", va="top",
             weight="bold")
    fig.text(0.01, 0.90, n, fontsize=9, color=MUTED, ha="left", va="top")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=S0, label="at 1% false positives"), Patch(color=S1, label="at 0.1% false positives")],
              loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9, borderaxespad=0.2,
              labelcolor=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    paths = []
    for ext in ("png", "svg"):
        p = f"{out_stem}.{ext}"
        fig.savefig(p, facecolor=SURFACE, metadata={"Date": None} if ext == "svg" else None)
        paths.append(p)
    plt.close(fig)
    return paths
