import pandas as pd
import matplotlib.pyplot as plt

# Load parquet file
file_path = "reddit_detoxified.parquet"
df = pd.read_parquet(file_path)

# ---- Ensure required columns exist ----
required_cols = ["subreddit",
                 "chain_avg_old_toxicity", "chain_avg_new_toxicity"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ---- Group by subreddit ----
grouped = (
    df.groupby("subreddit")
    .agg(
        avg_old_toxicity=("chain_avg_old_toxicity", "mean"),
        avg_new_toxicity=("chain_avg_new_toxicity", "mean"),
        count=("subreddit", "size")
    )
    .reset_index()
)

# ---- Select top 10 subreddits by volume ----
top10 = grouped.sort_values(by="count", ascending=False).head(10)

# ---- Sort for better visualization ----
top10 = top10.sort_values(by="avg_old_toxicity", ascending=False)

# ---- Chart 1: Old Toxicity ----
plt.figure()
plt.barh(top10["subreddit"], top10["avg_old_toxicity"])
plt.xlabel("Average Old Toxicity")
plt.ylabel("Subreddit")
plt.title("Top 10 Subreddits - Old Toxicity")
plt.gca().invert_yaxis()
plt.show()

# ---- Chart 2: New Toxicity ----
plt.figure()
plt.barh(top10["subreddit"], top10["avg_new_toxicity"])
plt.xlabel("Average New Toxicity")
plt.ylabel("Subreddit")
plt.title("Top 10 Subreddits - New Toxicity")
plt.gca().invert_yaxis()
plt.show()

# ---- Chart 3: Improvement (Old - New) ----
top10["toxicity_reduction"] = top10["avg_old_toxicity"] - \
    top10["avg_new_toxicity"]

plt.figure()
plt.barh(top10["subreddit"], top10["toxicity_reduction"])
plt.xlabel("Toxicity Reduction")
plt.ylabel("Subreddit")
plt.title("Top 10 Subreddits - Toxicity Reduction")
plt.gca().invert_yaxis()
plt.show()

# ---- Optional: Save results ----
top10.to_csv("top10_subreddit_toxicity_analysis.csv", index=False)

print(top10)
