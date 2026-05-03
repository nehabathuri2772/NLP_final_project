"""
A3. Average Cosine Similarity Analysis
=======================================
Handles two dataset layouts automatically:

  LAYOUT A — Nested (your actual dataset):
    Top-level columns: subreddit, chain_cosine_similarity_avg, comments (list of dicts)
    Each comment dict contains: cosine_similarity, comment_body, toxic, etc.
    -> Script explodes comments to get per-comment cosine_similarity.

  LAYOUT B — Flat (already-exploded dataset):
    cosine_similarity is already a top-level column.
    -> Script uses it directly.

Outputs (separate figures):
  Plot 1 — Per-subreddit avg cosine similarity bar chart (sorted, colour-coded)
  Plot 2 — Overall cosine similarity distribution (histogram + KDE)
  Plot 3 — Per-subreddit box plot (spread / IQR)
  Plot 4 — Top-N vs Bottom-N subreddits comparison
  Plot 5 — Chain-level avg cosine similarity per subreddit (if chain col present)

Framework-agnostic: swap model/dataset by editing CONFIG only.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# =============================================================
# CONFIGURATION  <- only section you need to edit per experiment
# =============================================================
CONFIG = {
    # -- Data -------------------------------------------------
    #"data_path": "reddit_detoxified.parquet",   # .parquet or .csv
    "data_path": "LLM_reddit_detoxified.parquet",

    # -- Column names -----------------------------------------
    # Top-level subreddit column (always present)
    "subreddit_col": "subreddit",

    # Per-comment cosine similarity at top level.
    # Set to None if not present;
    "cosine_col": "cosine_similarity",

    # Nested comments column name (used when cosine_col is missing at top level)
    "comments_col": "comments",

    # Key inside each comment dict that holds cosine similarity
    "comment_cosine_key": "cosine_similarity",

    # "chain_cosine_col": "chain_cosine_similarity_avg",
    "chain_cosine_col": "chain_cosine_similarity_avg",

    # -- Experiment label -------------------------------------
    "model_name": "LLM",           # shown in all plot titles

    # -- Filters & display ------------------------------------
    "min_comments": 10,             # min comments per subreddit to include
    "top_n": 10,                    # subreddits shown in best/worst chart

    # -- Output -----------------------------------------------
    "output_dir": "figures",
    "dpi": 150,
}


# =============================================================
# DATA LOADING
# =============================================================

def load_data(path):
    """Load .parquet or .csv/.tsv. Parquet requires: pip install pyarrow"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            raise ImportError(
                "Reading .parquet requires pyarrow.\n"
                "Install with:  pip install pyarrow"
            )
    elif ext in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .parquet, .csv, or .tsv")


def explode_comments(df, cfg):
    """
    Explode the nested comments column into one row per comment,
    then unpack comment-level fields as top-level columns.

    Works whether each cell is:
      - a list of dicts      (normal pyarrow/parquet load)
      - a stringified list   (CSV exports)
    """
    comments_col = cfg["comments_col"]
    subreddit_col = cfg["subreddit_col"]
    cosine_key = cfg["comment_cosine_key"]

    if comments_col not in df.columns:
        raise ValueError(
            f"Comments column '{comments_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Parse stringified lists if needed (e.g. exported via CSV)
    sample = df[comments_col].dropna().iloc[0]
    if isinstance(sample, str):
        import json, ast
        df = df.copy()
        def parse_comment_str(x):
            if not isinstance(x, str):
                return x
            try:
                return json.loads(x)          # handles true/false/null
            except (json.JSONDecodeError, ValueError):
                return ast.literal_eval(x)    # fallback for Python repr
        df[comments_col] = df[comments_col].apply(parse_comment_str)

    # Keep subreddit + comments + any chain-level cols
    keep_cols = [subreddit_col, comments_col]
    chain_col = cfg.get("chain_cosine_col")
    if chain_col and chain_col in df.columns:
        keep_cols.append(chain_col)
    keep_cols = list(dict.fromkeys(keep_cols))

    df_exp = (
        df[keep_cols]
        .explode(comments_col)
        .reset_index(drop=True)
    )

    # Unpack comment dicts into columns
    comment_data = pd.json_normalize(df_exp[comments_col].dropna()).reset_index(drop=True)
    df_exp = df_exp.drop(columns=[comments_col]).reset_index(drop=True)
    df_flat = pd.concat([df_exp, comment_data], axis=1)

    if cosine_key not in df_flat.columns:
        raise ValueError(
            f"Key '{cosine_key}' not found inside comments.\n"
            f"Available comment keys: {comment_data.columns.tolist()}"
        )
    return df_flat


def prepare_data(df, cfg):
    """
    Auto-detect layout (nested vs flat) and return:
      (df_flat, cosine_col_name)
    """
    subreddit_col = cfg["subreddit_col"]
    cosine_col = cfg["cosine_col"]

    if subreddit_col not in df.columns:
        raise ValueError(
            f"Subreddit column '{subreddit_col}' not found.\n"
            f"Available: {df.columns.tolist()}"
        )

    # LAYOUT B: cosine_similarity already at top level
    if cosine_col and cosine_col in df.columns:
        print(f"     Layout: FLAT  (cosine col = '{cosine_col}')")
        return df.dropna(subset=[cosine_col]).copy(), cosine_col

    # LAYOUT A: nested comments need exploding
    print(f"     Layout: NESTED — exploding '{cfg['comments_col']}' ...")
    df_flat = explode_comments(df, cfg)
    actual_col = cfg["comment_cosine_key"]
    df_flat = df_flat.dropna(subset=[actual_col]).copy()
    print(f"     Exploded to {len(df_flat):,} comment rows")
    return df_flat, actual_col


# =============================================================
# STATISTICS
# =============================================================

def compute_subreddit_stats(df, subreddit_col, cosine_col, min_n):
    stats = (
        df.groupby(subreddit_col)[cosine_col]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
        .rename(columns={subreddit_col: "subreddit"})
    )
    stats = stats[stats["count"] >= min_n].sort_values("mean", ascending=False).reset_index(drop=True)
    stats["rank"] = stats.index + 1
    return stats


def compute_overall_stats(series):
    return {
        "count":  len(series),
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "min":    series.min(),
        "max":    series.max(),
        "q25":    series.quantile(0.25),
        "q75":    series.quantile(0.75),
    }


# =============================================================
# PLOT 1 — Per-subreddit sorted bar chart
# =============================================================

def plot_subreddit_bar(stats, overall, cfg, out_dir):
    fig_h = max(6, len(stats) * 0.38)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    mean_line = overall["mean"]
    colours = ["#2ecc71" if v >= mean_line else "#e74c3c" for v in stats["mean"]]
    bars = ax.barh(stats["subreddit"], stats["mean"],
                   color=colours, edgecolor="white", linewidth=0.4, height=0.72)
    ax.axvline(mean_line, color="#2c3e50", linewidth=1.5, linestyle="--")
    ax.errorbar(stats["mean"], stats["subreddit"],
                xerr=stats["std"].fillna(0), fmt="none",
                color="#555", linewidth=0.8, capsize=2)

    for bar, val in zip(bars, stats["mean"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7.5)

    ax.set_xlabel("Average Cosine Similarity", fontsize=12)
    ax.set_title(
        f"[{cfg['model_name']}]  Average Cosine Similarity by Subreddit\n"
        f"(n >= {cfg['min_comments']} comments per subreddit, sorted descending)",
        fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()
    ax.legend(handles=[
        mpatches.Patch(color="#2ecc71", label="Above overall mean"),
        mpatches.Patch(color="#e74c3c", label="Below overall mean"),
        Line2D([0], [0], color="#2c3e50", lw=1.5, ls="--",
               label=f"Overall mean = {mean_line:.4f}"),
    ], fontsize=9, loc="lower right")

    plt.tight_layout()
    _save(fig, out_dir, "A3_plot1_subreddit_bar.png", cfg["dpi"])


# =============================================================
# PLOT 2 — Overall distribution histogram + KDE
# =============================================================

def plot_overall_distribution(vals, overall, cfg, out_dir):
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=60, density=True, color="#3498db", alpha=0.55,
            edgecolor="white", linewidth=0.3, label="Frequency (density)")

    kde = gaussian_kde(vals, bw_method="scott")
    x = np.linspace(vals.min(), vals.max(), 500)
    ax.plot(x, kde(x), color="#2c3e50", linewidth=2, label="KDE")
    ax.axvline(overall["mean"],   color="#e74c3c", lw=1.8, ls="--",
               label=f"Mean   = {overall['mean']:.4f}")
    ax.axvline(overall["median"], color="#27ae60", lw=1.8, ls=":",
               label=f"Median = {overall['median']:.4f}")

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"[{cfg['model_name']}]  Overall Cosine Similarity Distribution\n"
        f"n = {overall['count']:,} comments",
        fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.text(0.02, 0.97,
            f"Mean:   {overall['mean']:.4f}\nMedian: {overall['median']:.4f}\n"
            f"Std:    {overall['std']:.4f}\nQ25:    {overall['q25']:.4f}\n"
            f"Q75:    {overall['q75']:.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#ccc", alpha=0.9))

    plt.tight_layout()
    _save(fig, out_dir, "A3_plot2_overall_distribution.png", cfg["dpi"])


# =============================================================
# PLOT 3 — Per-subreddit box plot
# =============================================================

def plot_boxplot(df, stats, subreddit_col, cosine_col, cfg, out_dir):
    valid_subs = stats["subreddit"].tolist()
    df_f = df[df[subreddit_col].isin(valid_subs)].copy()
    order = (
        df_f.groupby(subreddit_col)[cosine_col]
        .median().sort_values(ascending=False).index.tolist()
    )

    fig_h = max(6, len(order) * 0.42)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    data_by_sub = [df_f[df_f[subreddit_col] == s][cosine_col].values for s in order]

    ax.boxplot(data_by_sub, vert=False, patch_artist=True, labels=order,
               flierprops=dict(marker="o", markersize=2, alpha=0.4,
                               markerfacecolor="#e74c3c", markeredgewidth=0),
               medianprops=dict(color="#c0392b", linewidth=1.8),
               boxprops=dict(facecolor="#aed6f1", linewidth=0.8),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8))

    overall_mean = df_f[cosine_col].mean()
    ax.axvline(overall_mean, color="#2c3e50", lw=1.5, ls="--",
               label=f"Overall mean = {overall_mean:.4f}")
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_title(
        f"[{cfg['model_name']}]  Cosine Similarity Distribution per Subreddit\n"
        f"(ordered by median, n >= {cfg['min_comments']})",
        fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    _save(fig, out_dir, "A3_plot3_boxplot_per_subreddit.png", cfg["dpi"])


# =============================================================
# PLOT 4 — Top-N vs Bottom-N comparison
# =============================================================

def plot_top_bottom(stats, cfg, out_dir):
    n = min(cfg["top_n"], len(stats) // 2)
    top    = stats.head(n).copy()
    bottom = stats.tail(n).copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n * 0.55)))
    for ax, subset, colour, label in [
        (axes[0], top,    "#2ecc71", f"Top {n} — Highest Context Retention"),
        (axes[1], bottom, "#e74c3c", f"Bottom {n} — Lowest Context Retention"),
    ]:
        ax.barh(subset["subreddit"], subset["mean"],
                color=colour, edgecolor="white", linewidth=0.4)
        ax.errorbar(subset["mean"], subset["subreddit"],
                    xerr=subset["std"].fillna(0),
                    fmt="none", color="#555", linewidth=0.8, capsize=2)
        ax.set_xlabel("Average Cosine Similarity")
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.set_xlim(0, 1.08)
        ax.invert_yaxis()
        subs_list = subset["subreddit"].tolist()
        for _, row in subset.iterrows():
            ax.text(row["mean"] + 0.005, subs_list.index(row["subreddit"]),
                    f"{row['mean']:.4f}  (n={int(row['count'])})",
                    va="center", fontsize=8)

    fig.suptitle(f"[{cfg['model_name']}]  Best vs Worst Subreddits — Cosine Similarity",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out_dir, "A3_plot4_top_bottom_subreddits.png", cfg["dpi"])


# =============================================================
# PLOT 5 — Chain-level cosine similarity (optional)
# =============================================================

def plot_chain_cosine(df_orig, cfg, out_dir):
    """Uses pre-exploded df. Skipped if chain_cosine_col absent."""
    chain_col = cfg.get("chain_cosine_col")
    sub_col   = cfg["subreddit_col"]

    if not chain_col or chain_col not in df_orig.columns:
        print("     (Skipping Plot 5 — chain cosine col not found)")
        return

    df_c = df_orig.dropna(subset=[chain_col]).copy()
    chain_stats = (
        df_c.groupby(sub_col)[chain_col]
        .agg(count="count", mean="mean", std="std")
        .reset_index()
        .rename(columns={sub_col: "subreddit"})
        .query(f"count >= {cfg['min_comments']}")
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
    )

    overall_chain_mean = df_c[chain_col].mean()
    fig_h = max(6, len(chain_stats) * 0.38)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    colours = ["#8e44ad" if v >= overall_chain_mean else "#e67e22"
               for v in chain_stats["mean"]]
    bars = ax.barh(chain_stats["subreddit"], chain_stats["mean"],
                   color=colours, edgecolor="white", linewidth=0.4, height=0.72)
    ax.errorbar(chain_stats["mean"], chain_stats["subreddit"],
                xerr=chain_stats["std"].fillna(0),
                fmt="none", color="#555", linewidth=0.8, capsize=2)
    ax.axvline(overall_chain_mean, color="#2c3e50", lw=1.5, ls="--",
               label=f"Overall chain mean = {overall_chain_mean:.4f}")

    for bar, val in zip(bars, chain_stats["mean"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7.5)

    ax.legend(handles=[
        mpatches.Patch(color="#8e44ad", label="Above overall mean"),
        mpatches.Patch(color="#e67e22", label="Below overall mean"),
        Line2D([0], [0], color="#2c3e50", lw=1.5, ls="--",
               label=f"Overall chain mean = {overall_chain_mean:.4f}"),
    ], fontsize=9, loc="lower right")
    ax.set_xlabel("Chain-level Average Cosine Similarity", fontsize=12)
    ax.set_title(
        f"[{cfg['model_name']}]  Chain-level Avg Cosine Similarity by Subreddit\n"
        f"(conversation-thread average, n >= {cfg['min_comments']})",
        fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()

    plt.tight_layout()
    _save(fig, out_dir, "A3_plot5_chain_cosine_per_subreddit.png", cfg["dpi"])


# =============================================================
# HELPERS
# =============================================================

def _save(fig, out_dir, filename, dpi):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_stats_report(stats, overall, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "A3_subreddit_cosine_stats.csv")
    stats.to_csv(csv_path, index=False)
    print(f"  Saved stats: {csv_path}")

    print("\n" + "=" * 62)
    print(f"  A3 -- COSINE SIMILARITY SUMMARY  [{cfg['model_name']}]")
    print("=" * 62)
    print(f"  Total comments  : {overall['count']:,}")
    print(f"  Subreddits      : {len(stats)}")
    print(f"  Mean            : {overall['mean']:.4f}")
    print(f"  Median          : {overall['median']:.4f}")
    print(f"  Std             : {overall['std']:.4f}")
    print(f"  [Q25, Q75]      : [{overall['q25']:.4f}, {overall['q75']:.4f}]")
    print("-" * 62)
    print("  TOP 5 (best context retention):")
    for _, row in stats.head(5).iterrows():
        print(f"    #{int(row['rank']):>2}  r/{row['subreddit']:<28} mean={row['mean']:.4f}  n={int(row['count'])}")
    print("-" * 62)
    print("  BOTTOM 5 (worst context retention):")
    for _, row in stats.tail(5).iterrows():
        print(f"    #{int(row['rank']):>2}  r/{row['subreddit']:<28} mean={row['mean']:.4f}  n={int(row['count'])}")
    print("=" * 62 + "\n")


# =============================================================
# MAIN
# =============================================================

def run_analysis(cfg=CONFIG):
    """
    Run the full A3 pipeline.

    Swap models/datasets by passing a modified cfg:

        run_analysis({
            **CONFIG,
            "data_path":          "reddit_llama.parquet",
            "model_name":         "LLaMA-3",
            "cosine_col":         "cosine_similarity",
            "comment_cosine_key": "cosine_similarity",
            "chain_cosine_col":   "chain_cosine_similarity_avg",
        })
    """
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"\n[A3] Loading: {cfg['data_path']}")
    df_orig = load_data(cfg["data_path"])
    print(f"     {len(df_orig):,} rows x {len(df_orig.columns)} cols")
    print(f"     Columns: {df_orig.columns.tolist()}")

    df_flat, cosine_col = prepare_data(df_orig, cfg)
    print(f"     Ready: {len(df_flat):,} comment rows | cosine col = '{cosine_col}'")

    print("[A3] Computing stats ...")
    stats   = compute_subreddit_stats(df_flat, cfg["subreddit_col"], cosine_col, cfg["min_comments"])
    overall = compute_overall_stats(df_flat[cosine_col])

    print("[A3] Generating plots ...")
    plot_subreddit_bar(stats, overall, cfg, cfg["output_dir"])
    plot_overall_distribution(df_flat[cosine_col].values, overall, cfg, cfg["output_dir"])
    plot_boxplot(df_flat, stats, cfg["subreddit_col"], cosine_col, cfg, cfg["output_dir"])
    plot_top_bottom(stats, cfg, cfg["output_dir"])
    plot_chain_cosine(df_orig, cfg, cfg["output_dir"])

    save_stats_report(stats, overall, cfg, cfg["output_dir"])
    print(f"[A3] Done -> {cfg['output_dir']}/\n")
    return stats, overall


if __name__ == "__main__":
    stats, overall = run_analysis()