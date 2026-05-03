import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = "reddit_detoxified.parquet"

OLD_COL = "chain_avg_old_toxicity"
NEW_COL = "chain_avg_new_toxicity"
GROUP_COL = "subreddit"

MODEL_NAME = "Baseline Model"
TOP_N = 10

# Load
df = pd.read_parquet(FILE_PATH)

print("Loaded data shape:", df.shape)
print("Columns:", df.columns)

# Normalize column names
df.columns = df.columns.str.strip()

# Check required columns
required_cols = [GROUP_COL, OLD_COL, NEW_COL]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Clean
df = df[required_cols].dropna()
print("After cleaning:", df.shape)

# Overall
old_avg = df[OLD_COL].mean()
new_avg = df[NEW_COL].mean()

overall_pct_change = ((new_avg - old_avg) / (old_avg + 1e-8)) * 100

print(f"\n=== {MODEL_NAME} ===")
print(f"Old Avg Toxicity: {old_avg:.4f}")
print(f"New Avg Toxicity: {new_avg:.4f}")
print(f"Overall % Change: {overall_pct_change:.2f}%")

# Group
grouped = df.groupby(GROUP_COL).agg({
    OLD_COL: "mean",
    NEW_COL: "mean"
}).reset_index()

grouped["pct_change"] = (
    (grouped[NEW_COL] - grouped[OLD_COL]) /
    (grouped[OLD_COL] + 1e-8)
) * 100

# Top N
topN = grouped.sort_values(by="pct_change").head(TOP_N)

# Plot 1
plt.figure(figsize=(12, 6))
x = range(len(topN))

plt.bar(x, topN[OLD_COL], width=0.4, label="Old")
plt.bar([i + 0.4 for i in x], topN[NEW_COL], width=0.4, label="New")

plt.xticks([i + 0.2 for i in x], topN[GROUP_COL], rotation=60, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2
plt.figure(figsize=(12, 6))
sns.barplot(data=topN, x=GROUP_COL, y="pct_change")
plt.axhline(0)
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.show()

# Plot 3
plt.figure(figsize=(5, 5))
plt.bar(["Overall"], [overall_pct_change])
plt.tight_layout()
plt.show()

# Save
safe_model_name = MODEL_NAME.replace(" ", "_")
topN.to_csv(f"{safe_model_name}_top{TOP_N}.csv", index=False)