"""
A2: Per-Comment Chain Average % Toxic
======================================
For every Reddit post (comment chain), calculates:
    chain_toxic_pct = chain_toxic_comment_count / len(comment_chain) * 100

Then averages that percentage across all chains — overall and by subreddit.

This answers: "If I visit a random Reddit post, what % of its comments
              do I expect to be toxic?"

Run:
    python A2_chain_toxic_pct.py

Outputs (written to ./outputs/):
    A2_overall.csv          — single-row overall summary
    A2_by_subreddit.csv     — per-subreddit breakdown (all chains)
    A2_charts.png           — all charts

Adapting to a different model or dataset:
    Only edit the CONFIG block below. The rest of the code never changes.
    Works with Qwen / LLaMA / Gemini output, Reddit or any other dataset,
    parquet or CSV input.
"""

from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — only edit this block to switch model / dataset
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Input ────────────────────────────────────────────────────────────────
    # Path to the parquet (or .csv) produced by your detox pipeline.
    "input_file": "reddit_detoxified.parquet",

    # ── Column names in the DataFrame ────────────────────────────────────────
    # Community / group column.
    # Reddit → "subreddit" | news → "category" | custom → whatever yours is.
    "group_col": "subreddit",

    # Column that holds the list/array of comment dicts per chain.
    "comments_col": "comments",

    # Column that stores the pre-computed toxic comment count per chain.
    # Set to None to force the script to count from scratch using
    # toxic_label_key / score_key + score_threshold below.
    "chain_toxic_count_col": "chain_toxic_comment_count",

    # ── Keys inside each comment dict (used when counting from scratch) ──────
    # Key for the pre-labeled boolean toxic flag (True / False).
    # Set to None if your pipeline only stores a raw numeric score.
    "toxic_label_key": "toxic",

    # Key for the raw toxicity score (float 0–1).
    # Fallback when toxic_label_key is None or missing on a comment.
    "score_key": "toxicity_score",

    # Score at/above which a comment is considered toxic.
    "score_threshold": 0.5,

    # ── Chart controls ───────────────────────────────────────────────────────
    # Minimum number of chains a subreddit must have to appear in charts.
    "min_chains_for_chart": 3,

    # How many subreddits to show in the grouped-bar chart (top by volume).
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
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Use .parquet or .csv")
    print(f"  Loaded {len(df):,} rows  |  "
          f"{df[cfg['group_col']].nunique():,} unique {cfg['group_col']}s")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Build chain-level table
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_toxic(comment, cfg):
    """
    Return True/False for a single comment dict.
    Priority: labeled bool field → score >= threshold → None (skip).
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


def _count_toxic_from_comments(comments, cfg):
    """
    Count toxic comments by iterating the list of comment dicts.
    Used when chain_toxic_count_col is None or NaN on a row.
    """
    count = 0
    for c in comments:
        if not isinstance(c, dict):
            continue
        t = _resolve_toxic(c, cfg)
        if t is True:
            count += 1
    return count


def build_chain_table(df, cfg):
    """
    Returns a DataFrame with one row per chain (Reddit post), containing:
        group               — subreddit (or whatever group_col is)
        chain_len           — total comments in the chain
        toxic_count         — number of toxic comments
        chain_toxic_pct     — toxic_count / chain_len * 100
    """
    pre_col = cfg["chain_toxic_count_col"]
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        group = row.get(cfg["group_col"], "unknown")
        comments = row.get(cfg["comments_col"])

        # Normalise comments to a plain Python list
        if comments is None:
            skipped += 1
            continue
        try:
            comments = list(comments)
        except TypeError:
            skipped += 1
            continue
        if not comments:
            continue  # empty chain — skip silently

        chain_len = len(comments)

        # Prefer pre-computed column; fall back to counting from scratch
        toxic_count = None
        if pre_col and pre_col in df.columns:
            raw = row.get(pre_col)
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                try:
                    toxic_count = int(raw)
                except (TypeError, ValueError):
                    pass

        if toxic_count is None:
            toxic_count = _count_toxic_from_comments(comments, cfg)

        chain_toxic_pct = (toxic_count / chain_len *
                           100) if chain_len > 0 else 0.0

        rows.append({
            "group":            str(group),
            "chain_len":        chain_len,
            "toxic_count":      toxic_count,
            "chain_toxic_pct":  round(chain_toxic_pct, 4),
        })

    if skipped:
        print(f"  Skipped {skipped:,} rows (missing/invalid comments column)")

    chains = pd.DataFrame(rows)
    print(f"  Chain table: {len(chains):,} chains  |  "
          f"{chains['group'].nunique():,} {cfg['group_col']}s")
    return chains


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Compute statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_overall(chains):
    """
    Overall average % toxic per chain.
    This is the expected % of toxic comments if you visit a random post.
    """
    total_chains = len(chains)
    total_comments = int(chains["chain_len"].sum())
    total_toxic = int(chains["toxic_count"].sum())

    return {
        "total_chains":              total_chains,
        "total_comments":            total_comments,
        "total_toxic_comments":      total_toxic,
        "avg_chain_toxic_pct":       round(chains["chain_toxic_pct"].mean(), 2),
        "median_chain_toxic_pct":    round(chains["chain_toxic_pct"].median(), 2),
        "std_chain_toxic_pct":       round(chains["chain_toxic_pct"].std(), 2),
        "min_chain_toxic_pct":       round(chains["chain_toxic_pct"].min(), 2),
        "max_chain_toxic_pct":       round(chains["chain_toxic_pct"].max(), 2),
        "avg_chain_len":             round(chains["chain_len"].mean(), 2),
    }


def compute_by_subreddit(chains, cfg):
    """
    Per-subreddit aggregation sorted by number of chains (descending).
    Filters out subreddits with fewer than cfg['min_chains_for_chart'] chains.
    """
    g = chains.groupby("group")

    result = pd.DataFrame({
        "num_chains":               g["chain_toxic_pct"].count(),
        "avg_chain_toxic_pct":      g["chain_toxic_pct"].mean().round(2),
        "median_chain_toxic_pct":   g["chain_toxic_pct"].median().round(2),
        "std_chain_toxic_pct":      g["chain_toxic_pct"].std().round(2),
        "avg_chain_len":            g["chain_len"].mean().round(2),
        "total_toxic_comments":     g["toxic_count"].sum().astype(int),
        "total_comments":           g["chain_len"].sum().astype(int),
    }).reset_index().rename(columns={"group": cfg["group_col"]})

    # Overall toxic % computed from raw totals (weighted, not mean-of-means)
    result["overall_toxic_pct"] = (
        result["total_toxic_comments"] / result["total_comments"] * 100
    ).round(2)

    result = result[result["num_chains"] >= cfg["min_chains_for_chart"]]
    result = result.sort_values(
        "num_chains", ascending=False).reset_index(drop=True)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Print report
# ══════════════════════════════════════════════════════════════════════════════

def print_report(overall, by_sub, cfg):
    sep = "═" * 70
    print(f"\n{sep}")
    print("  A2 — Per-Comment Chain Average % Toxic")
    print(sep)

    print(f"\n  ── OVERALL {'─' * 50}")
    print(f"  Total chains analysed       : {overall['total_chains']:>8,}")
    print(f"  Total comments              : {overall['total_comments']:>8,}")
    print(
        f"  Total toxic comments        : {overall['total_toxic_comments']:>8,}")
    print(
        f"  Avg chain toxic %           : {overall['avg_chain_toxic_pct']:>8.2f} %")
    print(
        f"  Median chain toxic %        : {overall['median_chain_toxic_pct']:>8.2f} %")
    print(
        f"  Std dev                     : {overall['std_chain_toxic_pct']:>8.2f} %")
    print(f"  Min / Max chain toxic %     : {overall['min_chain_toxic_pct']:>6.2f} % / "
          f"{overall['max_chain_toxic_pct']:.2f} %")
    print(
        f"  Avg chain length            : {overall['avg_chain_len']:>8.2f} comments")

    grp = cfg["group_col"]
    print(f"\n  ── BY {grp.upper()} "
          f"(≥ {cfg['min_chains_for_chart']} chains, {len(by_sub)} {grp}s shown) "
          f"{'─' * 15}")

    display = by_sub.rename(columns={
        grp:                       grp,
        "num_chains":              "chains",
        "avg_chain_toxic_pct":     "avg_%_toxic",
        "median_chain_toxic_pct":  "median_%_toxic",
        "avg_chain_len":           "avg_len",
        "total_toxic_comments":    "toxic_comments",
        "total_comments":          "total_comments",
        "overall_toxic_pct":       "weighted_%_toxic",
    })
    cols = [grp, "chains", "avg_%_toxic", "median_%_toxic",
            "avg_len", "toxic_comments", "total_comments", "weighted_%_toxic"]
    print(display[cols].to_string(index=False))
    print()
 # ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Charts
# ══════════════════════════════════════════════════════════════════════════════


TEAL = "#0A9396"
AMBER = "#EE9B00"
CORAL = "#D85A30"
NAVY = "#0D1B2A"
LGRAY = "#E9F2F3"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)


def chart_overall_summary(overall, ax):
    """Single horizontal bar showing avg % toxic per chain."""
    pct = overall["avg_chain_toxic_pct"]
    ax.barh([0], [pct],        color=AMBER, height=0.35,
            zorder=3, label=f"Avg toxic {pct:.1f}%")
    ax.barh([0], [100 - pct],  color=TEAL,  height=0.35, left=pct, zorder=3,
            label=f"Avg non-toxic {100 - pct:.1f}%")

    ax.text(pct / 2, 0, f"{pct:.1f}%",
            ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    ax.text(pct + (100 - pct) / 2, 0, f"{100 - pct:.1f}%",
            ha="center", va="center", fontsize=13, color="white", fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("% of comments per chain (average)", fontsize=10)
    ax.set_title(
        f"Overall — avg chain toxic %  "
        f"({overall['total_chains']:,} chains, {overall['total_comments']:,} comments)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def chart_by_subreddit_bar(by_sub, cfg, ax):
    """Grouped bar: avg chain toxic % per subreddit (top N by chain count)."""
    top = by_sub.head(cfg["top_n"]).copy()
    grp = cfg["group_col"]
    subs = top[grp].tolist()
    x = np.arange(len(subs))
    w = 0.38

    bt = ax.bar(x - w / 2, top["avg_chain_toxic_pct"],
                width=w, color=AMBER, label="Avg toxic %",     zorder=3)
    bm = ax.bar(x + w / 2, top["median_chain_toxic_pct"],
                width=w, color=TEAL,  label="Median toxic %",  zorder=3)

    for bar in list(bt) + list(bm):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.0f}", ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("% toxic comments per chain", fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_title(
        f"Avg & median chain toxic % per {grp}  (top {cfg['top_n']} by chain volume)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=False)

    # Chain count on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v} chains" for v in top["num_chains"]],
                        fontsize=7.5, color="#999")
    ax2.tick_params(length=0)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
    _style(ax)


def chart_ranked_avg(by_sub, cfg, ax):
    """
    Horizontal bar — ALL subreddits ranked by avg chain toxic %,
    colour-coded: orange for high toxicity, teal for low.
    """
    ranked = by_sub.sort_values("avg_chain_toxic_pct", ascending=True).copy()
    grp = cfg["group_col"]
    y = np.arange(len(ranked))
    vals = ranked["avg_chain_toxic_pct"].values
    colors = [AMBER if v >= ranked["avg_chain_toxic_pct"].mean()
              else TEAL for v in vals]
    h = max(0.28, min(0.60, 12 / max(len(ranked), 1)))

    ax.barh(y, vals, color=colors, height=h, zorder=3)

    mean_val = ranked["avg_chain_toxic_pct"].mean()
    ax.axvline(mean_val, color=CORAL, linewidth=1.8, linestyle="--",
               label=f"Overall mean {mean_val:.1f}%", zorder=4)

    for i, v in enumerate(vals):
        ax.text(v + 0.4, i, f"{v:.1f}%",
                va="center", fontsize=8, color="#444")

    ax.set_yticks(y)
    ax.set_yticklabels(ranked[grp], fontsize=8.5)
    ax.set_xlabel("Avg % of comments toxic per chain", fontsize=10)
    ax.set_title(
        f"All {grp}s — avg chain toxic % ranked  "
        f"(orange ≥ mean, teal < mean)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def chart_distribution(chains, overall, ax):
    """
    Histogram of chain_toxic_pct across all chains,
    with mean and median reference lines.
    """
    vals = chains["chain_toxic_pct"].values
    ax.hist(vals, bins=20, color=TEAL,
            edgecolor="white", linewidth=0.6, zorder=3)

    mean_v = overall["avg_chain_toxic_pct"]
    median_v = overall["median_chain_toxic_pct"]
    ax.axvline(mean_v,   color=AMBER, linewidth=2,   linestyle="--",
               label=f"Mean {mean_v:.1f}%",   zorder=4)
    ax.axvline(median_v, color=CORAL, linewidth=2,   linestyle=":",
               label=f"Median {median_v:.1f}%", zorder=4)

    ax.set_xlabel("Chain toxic % (per post)", fontsize=10)
    ax.set_ylabel("Number of chains", fontsize=10)
    ax.set_title("Distribution of chain toxic % across all posts",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    _style(ax)


def make_charts(chains, overall, by_sub, cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1. Overall summary chart
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    chart_overall_summary(overall, ax)
    plt.tight_layout()
    p1 = out_dir / "A2_overall_summary.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p1}")

    # -----------------------------
    # 2. Grouped bar by subreddit
    # -----------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    chart_by_subreddit_bar(by_sub, cfg, ax)
    plt.tight_layout()
    p2 = out_dir / "A2_by_subreddit_bar.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p2}")

    # -----------------------------
    # 3. Distribution histogram
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    chart_distribution(chains, overall, ax)
    plt.tight_layout()
    p3 = out_dir / "A2_distribution.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p3}")

    # -----------------------------
    # 4. Ranked avg toxic %
    # -----------------------------
    ranked_height = max(5, len(by_sub) * 0.38)

    fig, ax = plt.subplots(figsize=(12, ranked_height))
    chart_ranked_avg(by_sub, cfg, ax)
    plt.tight_layout()
    p4 = out_dir / "A2_ranked_avg.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p4}")

    print(f"\nAll A2 charts saved in → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Save CSVs
# ══════════════════════════════════════════════════════════════════════════════

def save_csvs(overall, by_sub, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "A2_overall.csv"
    pd.DataFrame([overall]).to_csv(p1, index=False)
    print(f"  CSV saved    → {p1}")

    p2 = out_dir / "A2_by_subreddit.csv"
    by_sub.to_csv(p2, index=False)
    print(f"  CSV saved    → {p2}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg=CONFIG):
    print("\n" + "─" * 70)
    print("  A2: Per-Comment Chain Average % Toxic")
    print("─" * 70)

    print("\n[1/5] Loading data...")
    df = load_data(cfg)

    print("[2/5] Building chain-level table...")
    chains = build_chain_table(df, cfg)
    if chains.empty:
        raise RuntimeError("No chains found. Check column names in CONFIG.")

    print("[3/5] Computing statistics...")
    overall = compute_overall(chains)
    by_sub = compute_by_subreddit(chains, cfg)

    print("[4/5] Printing report...")
    print_report(overall, by_sub, cfg)

    print("[5/5] Saving charts and CSVs...")
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    make_charts(chains, overall, by_sub, cfg, cfg["output_dir"])
    save_csvs(overall, by_sub, cfg["output_dir"])

    print("\nDone.\n")
    return overall, by_sub


if __name__ == "__main__":
    overall, by_subreddit = run(CONFIG)
