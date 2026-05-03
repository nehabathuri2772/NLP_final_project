"""
A1: Average Toxicity Improvement Analysis (Old vs New)
=====================================================

- Groups by subreddit
- Compares chain_avg_old_toxicity vs chain_avg_new_toxicity
- Works for any LLM output dataset
- Generates separate charts

Run:
    python A1_llm.py
"""

# ── Imports ─────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────
CONFIG = {
    "input_file": "PTSD_reddit_detoxified.parquet",  # UPDATED HERE
    "group_col": "subreddit",
    "old_col": "chain_avg_old_toxicity",
    "new_col": "chain_avg_new_toxicity",
    "top_n": 10,
    "output_dir": "outputs"
}

# ── Load Data ───────────────────────────────────────────


def load_data(cfg):
    path = Path(cfg["input_file"])

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format")

    print(f"Loaded {len(df)} rows from {path.name}")
    return df


# ── Compute Stats ───────────────────────────────────────
def compute_stats(df, cfg):
    grouped = (
        df.groupby(cfg["group_col"])
        .agg(
            avg_old=(cfg["old_col"], "mean"),
            avg_new=(cfg["new_col"], "mean"),
            count=(cfg["group_col"], "size")
        )
        .reset_index()
    )

    # Top N subreddits by volume
    top = grouped.sort_values(by="count", ascending=False).head(cfg["top_n"])

    # Compute improvement
    top["improvement"] = top["avg_old"] - top["avg_new"]

    return top.sort_values(by="avg_old", ascending=False)


# ── Charts ──────────────────────────────────────────────
def plot_old(top, out_dir):
    plt.figure()
    plt.barh(top["subreddit"], top["avg_old"])
    plt.xlabel("Average Old Toxicity")
    plt.title("Top PTSD Subreddits - OLD Toxicity")
    plt.gca().invert_yaxis()

    path = out_dir / "old_toxicity.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_new(top, out_dir):
    plt.figure()
    plt.barh(top["subreddit"], top["avg_new"])
    plt.xlabel("Average New Toxicity")
    plt.title("Top PTSD Subreddits - NEW Toxicity")
    plt.gca().invert_yaxis()

    path = out_dir / "new_toxicity.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_improvement(top, out_dir):
    plt.figure()
    plt.barh(top["subreddit"], top["improvement"])
    plt.xlabel("Toxicity Reduction (Old - New)")
    plt.title("Toxicity Improvement by Subreddit")
    plt.gca().invert_yaxis()

    path = out_dir / "toxicity_improvement.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ── Save CSV ────────────────────────────────────────────
def save_csv(top, out_dir):
    path = out_dir / "toxicity_summary.csv"
    top.to_csv(path, index=False)
    print(f"Saved → {path}")


# ── MAIN ────────────────────────────────────────────────
def run(cfg):
    print("\nRunning PTSD Toxicity Analysis...\n")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True)

    df = load_data(cfg)
    top = compute_stats(df, cfg)

    print("\nTop Subreddits:\n")
    print(top)

    plot_old(top, out_dir)
    plot_new(top, out_dir)
    plot_improvement(top, out_dir)
    save_csv(top, out_dir)

    print("\nDone.\n")


# ── ENTRY POINT ─────────────────────────────────────────
if __name__ == "__main__":
    run(CONFIG)
