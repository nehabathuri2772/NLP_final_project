"""
A2: Per-Comment Chain Average % Toxic
======================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import json

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "input_file": "LLM_reddit_detoxified.parquet",  # ✅ FIXED

    "group_col": "subreddit",
    "comments_col": "comments",

    "chain_toxic_count_col": "chain_toxic_comment_count",

    "toxic_label_key": "toxic",
    "score_key": "toxicity_score",
    "score_threshold": 0.5,

    "min_chains_for_chart": 3,
    "top_n": 10,

    "output_dir": "outputs",
}


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
def load_data(cfg):
    path = Path(cfg["input_file"])
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported format")

    print(f"Loaded {len(df):,} rows")

    # 🔍 Debug schema
    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ─────────────────────────────────────────────────────────────
# SAFE COMMENT PARSER
# ─────────────────────────────────────────────────────────────
def parse_comments(raw):
    """
    Handles:
    - list
    - numpy array
    - JSON string (common in LLM outputs)
    """
    if raw is None:
        return None

    if isinstance(raw, list):
        return raw

    # Handle numpy arrays
    try:
        return list(raw)
    except:
        pass

    # Handle JSON string
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except:
            return None

    return None


# ─────────────────────────────────────────────────────────────
# TOXIC RESOLUTION
# ─────────────────────────────────────────────────────────────
def resolve_toxic(comment, cfg):
    lk = cfg["toxic_label_key"]
    sk = cfg["score_key"]
    th = cfg["score_threshold"]

    if not isinstance(comment, dict):
        return None

    if lk and lk in comment:
        val = comment[lk]
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")

    if sk and sk in comment:
        try:
            return float(comment[sk]) >= th
        except:
            pass

    return None


def count_toxic(comments, cfg):
    count = 0
    for c in comments:
        if resolve_toxic(c, cfg) is True:
            count += 1
    return count


# ─────────────────────────────────────────────────────────────
# BUILD CHAINS
# ─────────────────────────────────────────────────────────────
def build_chain_table(df, cfg):
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        group = row.get(cfg["group_col"], "unknown")

        comments_raw = row.get(cfg["comments_col"])
        comments = parse_comments(comments_raw)

        if not comments:
            skipped += 1
            continue

        chain_len = len(comments)

        # Use precomputed if exists
        toxic_count = None
        col = cfg["chain_toxic_count_col"]

        if col in df.columns:
            val = row.get(col)
            if pd.notna(val):
                try:
                    toxic_count = int(val)
                except:
                    pass

        if toxic_count is None:
            toxic_count = count_toxic(comments, cfg)

        pct = (toxic_count / chain_len) * 100 if chain_len else 0

        rows.append({
            "group": str(group),
            "chain_len": chain_len,
            "toxic_count": toxic_count,
            "chain_toxic_pct": round(pct, 4),
        })

    print(f"Skipped {skipped} rows")

    chains = pd.DataFrame(rows)
    print(f"Built {len(chains):,} chains")

    return chains


# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────
def compute_overall(chains):
    return {
        "total_chains": len(chains),
        "total_comments": int(chains["chain_len"].sum()),
        "total_toxic_comments": int(chains["toxic_count"].sum()),
        "avg_chain_toxic_pct": round(chains["chain_toxic_pct"].mean(), 2),
        "median_chain_toxic_pct": round(chains["chain_toxic_pct"].median(), 2),
    }


def compute_by_subreddit(chains, cfg):
    g = chains.groupby("group")

    df = pd.DataFrame({
        "num_chains": g.size(),
        "avg_chain_toxic_pct": g["chain_toxic_pct"].mean().round(2),
        "median_chain_toxic_pct": g["chain_toxic_pct"].median().round(2),
    }).reset_index()

    return df.sort_values("num_chains", ascending=False)


# ─────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────
def plot(chains, overall, out_dir):
    plt.figure()
    plt.hist(chains["chain_toxic_pct"], bins=20)
    plt.axvline(overall["avg_chain_toxic_pct"], linestyle="--")
    plt.title("Distribution of Toxic % per Chain")

    path = Path(out_dir) / "A2_distribution.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run(cfg):
    print("\nRunning A2...\n")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True)

    df = load_data(cfg)

    chains = build_chain_table(df, cfg)

    if chains.empty:
        raise RuntimeError("No valid chains found — check comments column")

    overall = compute_overall(chains)
    by_sub = compute_by_subreddit(chains, cfg)

    print("\nOverall:")
    print(overall)

    print("\nTop subreddits:")
    print(by_sub.head(10))

    plot(chains, overall, out_dir)

    chains.to_csv(out_dir / "A2_chains.csv", index=False)
    by_sub.to_csv(out_dir / "A2_by_subreddit.csv", index=False)

    print("\nDone.\n")


if __name__ == "__main__":
    run(CONFIG)
