"""
A1: Toxic vs Non-toxic Comment Analysis — grouped by subreddit
==============================================================
Run:
    python A1_toxic_vs_nontoxic.py

Outputs (written to ./outputs/):
    A1_overall.csv          — single-row overall summary
    A1_by_subreddit.csv     — per-subreddit breakdown
    A1_charts.png           — pie + grouped bar + ranked stacked bar

Adapting to a different model or dataset:
    Only edit the CONFIG block below. Nothing else needs to change.
    The code works whether your pipeline uses:
      - Qwen / LLaMA / Gemini (or any other model)
      - A pre-labeled 'toxic' bool field  OR just a raw score column
      - A different column name for communities (category, domain, etc.)
      - Parquet or CSV input
"""

# ── stdlib / third-party ─────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — only edit this block to switch model / dataset
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Input ────────────────────────────────────────────────────────────────
    # Path to the parquet (or .csv) file produced by your detox pipeline.
    "input_file": "reddit_detoxified.parquet",

    # ── Column names in the DataFrame ────────────────────────────────────────
    # Column that holds the community / group name.
    # Reddit → "subreddit" | news → "category" | custom → whatever yours is.
    "group_col": "subreddit",

    # Column that holds the list/array of comment dicts for each post/chain.
    "comments_col": "comments",

    # ── Keys inside each comment dict ────────────────────────────────────────
    # Key for the pre-labeled boolean toxic flag (True / False).
    # Set to None if your pipeline only stores a raw numeric score.
    "toxic_label_key": "toxic",

    # Key for the raw toxicity score (float 0-1).
    # Used as a fallback when toxic_label_key is None or missing on a comment.
    "score_key": "toxicity_score",

    # ── Threshold ────────────────────────────────────────────────────────────
    # Score at/above which a comment is considered toxic.
    # Only applied when deriving the label from score_key.
    "score_threshold": 0.5,

    # ── Chart controls ───────────────────────────────────────────────────────
    # Minimum comments a subreddit must have to appear in the charts.
    "min_comments_for_chart": 10,

    # How many subreddits to show in the grouped-bar and ranked charts.
    "top_n": 10,

    # ── Output ───────────────────────────────────────────────────────────────
    "output_dir": "outputs",
}
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load
# ══════════════════════════════════════════════════════════════════════════════

def load_data(cfg):
    path = Path(cfg["input_file"])
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format '{path.suffix}'. Use .parquet or .csv")
    print(f"Loaded {len(df):,} rows, {df[cfg['group_col']].nunique():,} unique {cfg['group_col']}s")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Flatten comments into one row per comment
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_toxic(comment, cfg):
    """
    Return True/False for a single comment dict.
    Returns None if the label cannot be determined (comment is skipped).

    Logic:
      1. Use cfg["toxic_label_key"] if the key exists in the comment.
      2. Fall back to cfg["score_key"] >= cfg["score_threshold"].
      3. Return None if neither key is present.
    """
    lk = cfg["toxic_label_key"]
    sk = cfg["score_key"]
    th = cfg["score_threshold"]

    if lk and lk in comment:
        val = comment[lk]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes")

    if sk and sk in comment:
        try:
            return float(comment[sk]) >= th
        except (TypeError, ValueError):
            pass

    return None


def flatten_comments(df, cfg):
    """
    Iterate every row, iterate every comment inside the comments column,
    and build a flat DataFrame: one row per comment.

    Columns returned:
        subreddit        (or whatever group_col is)
        toxic            bool
        toxicity_score   float  (NaN when absent)
    """
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        group    = row.get(cfg["group_col"], "unknown")
        comments = row.get(cfg["comments_col"])

        # Tolerate None, empty lists, numpy arrays
        if comments is None:
            continue
        try:
            comments = list(comments)
        except TypeError:
            continue
        if not comments:
            continue

        for c in comments:
            if not isinstance(c, dict):
                skipped += 1
                continue
            toxic = _resolve_toxic(c, cfg)
            if toxic is None:
                skipped += 1
                continue

            score_key = cfg.get("score_key")
            score = float(c[score_key]) if score_key and score_key in c else float("nan")

            rows.append({
                "subreddit":      str(group),
                "toxic":          bool(toxic),
                "toxicity_score": score,
            })

    if skipped:
        print(f"  Skipped {skipped:,} comments (no usable label or score)")

    flat = pd.DataFrame(rows)
    print(f"  Flat comment table: {len(flat):,} comments across "
          f"{flat['subreddit'].nunique():,} {cfg['group_col']}s")
    return flat


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Compute statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_overall(flat):
    total    = len(flat)
    n_toxic  = int(flat["toxic"].sum())
    n_clean  = total - n_toxic
    return {
        "total_comments":     total,
        "toxic_count":        n_toxic,
        "nontoxic_count":     n_clean,
        "toxic_pct":          round(100 * n_toxic / total, 2) if total else 0.0,
        "nontoxic_pct":       round(100 * n_clean / total, 2) if total else 0.0,
        "avg_toxicity_score": round(flat["toxicity_score"].mean(), 4)
        # "avg_toxicity_score": round(flat["chain_avg_old_toxicity"].mean(), 4),
    }


def compute_by_subreddit(flat, cfg):
    """
    Returns a DataFrame with one row per subreddit, sorted by total comments desc.
    Filters out subreddits with fewer than cfg["min_comments_for_chart"] comments.
    """
    g = flat.groupby("subreddit")

    result = pd.DataFrame({
        "total_comments":     g["toxic"].count(),
        "toxic_count":        g["toxic"].sum().astype(int),
        "avg_toxicity_score": g["toxicity_score"].mean().round(4),
    }).reset_index()

    result["nontoxic_count"] = result["total_comments"] - result["toxic_count"]
    result["toxic_pct"]      = (100 * result["toxic_count"] / result["total_comments"]).round(2)
    result["nontoxic_pct"]   = (100 - result["toxic_pct"]).round(2)

    result = result[result["total_comments"] >= cfg["min_comments_for_chart"]]
    result = result.sort_values("total_comments", ascending=False).reset_index(drop=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Print report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(overall, by_sub, cfg):
    sep = "═" * 65
    print(f"\n{sep}")
    print("  A1 — Toxic vs Non-toxic Comment Breakdown")
    print(sep)

    print(f"\n  ── OVERALL {'─' * 45}")
    print(f"  Total comments   : {overall['total_comments']:>8,}")
    print(f"  Toxic            : {overall['toxic_count']:>8,}   ({overall['toxic_pct']:.1f} %)")
    print(f"  Non-toxic        : {overall['nontoxic_count']:>8,}   ({overall['nontoxic_pct']:.1f} %)")
    print(f"  Avg score        : {overall['avg_toxicity_score']:>8.4f}")

    print(f"\n  ── BY SUBREDDIT (all {len(by_sub)} with ≥ {cfg['min_comments_for_chart']} comments) {'─' * 10}")
    display = by_sub.rename(columns={
        "subreddit":          "subreddit",
        "total_comments":     "total",
        "toxic_count":        "toxic",
        "nontoxic_count":     "non_toxic",
        "toxic_pct":          "toxic_%",
        "nontoxic_pct":       "nontoxic_%",
        "avg_toxicity_score": "avg_score",
    })
    print(display.to_string(index=False))
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Charts
# ══════════════════════════════════════════════════════════════════════════════

TEAL   = "#0A9396"
AMBER  = "#EE9B00"
NAVY   = "#0D1B2A"
LGRAY  = "#E9F2F3"
DGRAY  = "#3A5A60"


def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)


def chart_pie(overall, ax):
    sizes  = [overall["toxic_pct"], overall["nontoxic_pct"]]
    colors = [AMBER, TEAL]
    wedges, _, autotexts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        autopct="%1.1f%%",
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Overall — Toxic vs Non-toxic", fontsize=13, fontweight="bold", pad=16)
    patches = [
        mpatches.Patch(color=AMBER, label=f"Toxic  {overall['toxic_pct']}%  ({overall['toxic_count']:,})"),
        mpatches.Patch(color=TEAL,  label=f"Non-toxic  {overall['nontoxic_pct']}%  ({overall['nontoxic_count']:,})"),
    ]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.14), fontsize=10, frameon=False, ncol=1)


def chart_grouped_bar(by_sub, cfg, ax):
    top  = by_sub.head(cfg["top_n"]).copy()
    subs = top["subreddit"].tolist()
    x    = np.arange(len(subs))
    w    = 0.38

    bt = ax.bar(x - w / 2, top["toxic_pct"],    width=w, color=AMBER, label="Toxic",     zorder=3)
    bn = ax.bar(x + w / 2, top["nontoxic_pct"], width=w, color=TEAL,  label="Non-toxic", zorder=3)

    for bar in list(bt) + list(bn):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.7,
                f"{h:.0f}", ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("% of comments", fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_title(
        f"Toxic vs Non-toxic % per subreddit  (top {cfg['top_n']} by volume)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False)
    _style_ax(ax)

    # Total-comment count on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"n={v:,}" for v in top["total_comments"]],
                         fontsize=7.5, color="#999")
    ax2.tick_params(length=0)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)


def chart_stacked_ranked(by_sub, cfg, ax):
    """
    Horizontal 100 % stacked bar, subreddits ranked by toxic % (highest at top).
    Shows ALL subreddits that pass the min-comments filter, not just top_n.
    """
    ranked = by_sub.sort_values("toxic_pct", ascending=True).copy()
    y      = np.arange(len(ranked))
    h      = max(0.3, min(0.65, 12 / max(len(ranked), 1)))   # adaptive bar height

    ax.barh(y, ranked["toxic_pct"],    height=h, color=AMBER, label="Toxic",     zorder=3)
    ax.barh(y, ranked["nontoxic_pct"], height=h, color=TEAL,  label="Non-toxic",
            left=ranked["toxic_pct"], zorder=3)

    for i, (_, row) in enumerate(ranked.iterrows()):
        ax.text(0.6, i, f"{row['toxic_pct']:.1f}%",
                va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(ranked["subreddit"], fontsize=8.5)
    ax.set_xlabel("% of comments", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title(
        f"All subreddits — stacked toxic share (ranked highest to lowest)",
        fontsize=12, fontweight="bold",
    )
    patches = [
        mpatches.Patch(color=AMBER, label="Toxic"),
        mpatches.Patch(color=TEAL,  label="Non-toxic"),
    ]
    ax.legend(handles=patches, fontsize=9, frameon=False, loc="lower right")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def chart_avg_score(by_sub, cfg, ax):
    """Bar chart of average toxicity score per subreddit (top_n by volume)."""
    top  = by_sub.head(cfg["top_n"]).copy().sort_values("avg_toxicity_score", ascending=False)
    x    = np.arange(len(top))

    bars = ax.bar(x, top["avg_toxicity_score"], color=TEAL, width=0.6, zorder=3)
    ax.axhline(top["avg_toxicity_score"].mean(), color=AMBER, linewidth=1.5,
               linestyle="--", label=f"Mean {top['avg_toxicity_score'].mean():.3f}", zorder=4)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(top["subreddit"], rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("Avg toxicity score", fontsize=10)
    ax.set_title(f"Avg toxicity score per subreddit  (top {cfg['top_n']} by volume)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    _style_ax(ax)


def make_charts_old(overall, by_sub, cfg, out_dir):
    n_ranked = len(by_sub)
    ranked_height = max(6, n_ranked * 0.38)   # scales with number of subreddits

    fig = plt.figure(figsize=(20, 10 + ranked_height), facecolor="white")
    fig.suptitle("A1 — Toxic vs Non-toxic Comment Analysis",
                 fontsize=17, fontweight="bold", y=0.98)

    # Row 0: pie + grouped bar
    # Row 1: avg-score bar
    # Row 2: full ranked stacked bar (tall)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[5, 5, ranked_height],
        hspace=0.55, wspace=0.35,
    )

    ax_pie   = fig.add_subplot(gs[0, 0])
    ax_grp   = fig.add_subplot(gs[0, 1])
    ax_score = fig.add_subplot(gs[1, :])
    ax_stk   = fig.add_subplot(gs[2, :])

    chart_pie(overall, ax_pie)
    chart_grouped_bar(by_sub, cfg, ax_grp)
    chart_avg_score(by_sub, cfg, ax_score)
    chart_stacked_ranked(by_sub, cfg, ax_stk)

    # Footer with run metadata
    fig.text(
        0.5, 0.005,
        f"Source: {Path(cfg['input_file']).name}  |  "
        f"Toxicity threshold: {cfg['score_threshold']}  |  "
        f"Min comments per subreddit: {cfg['min_comments_for_chart']}  |  "
        f"Total comments: {overall['total_comments']:,}",
        ha="center", fontsize=9, color="#aaa",
    )

    out_path = Path(out_dir) / "A1_charts.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out_path}")

def make_charts(overall, by_sub, cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 1. Pie Chart
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    chart_pie(overall, ax)
    plt.tight_layout()
    pie_path = out_dir / "A1_pie_chart.png"
    plt.savefig(pie_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Pie chart saved      → {pie_path}")

    # -------------------------------
    # 2. Grouped Bar Chart
    # -------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    chart_grouped_bar(by_sub, cfg, ax)
    plt.tight_layout()
    grouped_path = out_dir / "A1_grouped_bar_chart.png"
    plt.savefig(grouped_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grouped bar saved    → {grouped_path}")

    # -------------------------------
    # 3. Average Score Chart
    # -------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    chart_avg_score(by_sub, cfg, ax)
    plt.tight_layout()
    avg_path = out_dir / "A1_avg_score_chart.png"
    plt.savefig(avg_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Avg score chart saved → {avg_path}")

    # -------------------------------
    # 4. Ranked Stacked Chart
    # -------------------------------
    ranked_height = max(6, len(by_sub) * 0.38)

    fig, ax = plt.subplots(
        figsize=(14, ranked_height),
        facecolor="white"
    )
    chart_stacked_ranked(by_sub, cfg, ax)
    plt.tight_layout()
    stacked_path = out_dir / "A1_ranked_stacked_chart.png"
    plt.savefig(stacked_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Ranked stacked saved → {stacked_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Save CSVs
# ══════════════════════════════════════════════════════════════════════════════

def save_csvs(overall, by_sub, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_path = out_dir / "A1_overall.csv"
    pd.DataFrame([overall]).to_csv(overall_path, index=False)
    print(f"  CSV saved   → {overall_path}")

    sub_path = out_dir / "A1_by_subreddit.csv"
    by_sub.to_csv(sub_path, index=False)
    print(f"  CSV saved   → {sub_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg=CONFIG):
    print("\n" + "─" * 65)
    print("  A1: Toxic vs Non-toxic Analysis")
    print("─" * 65)

    # 1. Load
    print("\n[1/5] Loading data...")
    df = load_data(cfg)

    # 2. Flatten
    print("[2/5] Flattening comment chains...")
    flat = flatten_comments(df, cfg)
    if flat.empty:
        raise RuntimeError("No comments found. Check column names in CONFIG.")

    # 3. Stats
    print("[3/5] Computing statistics...")
    overall = compute_overall(flat)
    by_sub  = compute_by_subreddit(flat, cfg)

    # 4. Report
    print("[4/5] Printing report...")
    print_report(overall, by_sub, cfg)

    # 5. Charts + CSVs
    print("[5/5] Saving charts and CSVs...")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # make charts call
    make_charts(overall, by_sub, cfg, cfg["output_dir"])
    save_csvs(overall, by_sub, cfg["output_dir"])

    print("\nDone.\n")
    return overall, by_sub


if __name__ == "__main__":
    overall, by_subreddit = run(CONFIG)
