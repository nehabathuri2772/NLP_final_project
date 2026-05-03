import pyarrow.parquet as pq
import pandas as pd
import re

# helper functions
def clean_text(text):
    """
    removes url and special chars
    returns clean text
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

#Read only 10,000 rows — no memory issues
parquet_file = pq.ParquetFile("RC_2012-02.parquet")
batch = next(parquet_file.iter_batches(batch_size=10_000))
df = batch.to_pandas()

# drop unwanted columns
df = df.drop(columns=['controversiality'])

#Drop deleted/empty bodies
df = df[df["body"].notna()]

# Remove deleted/removed comments
df = df[~df["body"].isin(["[deleted]", "[removed]"])]

# remove start and end space
df = df[df["body"].str.strip() != ""]

# keep more than 20
df = df[df["body"].str.len() >= 20]

# truncate long comments
df["body"] = df["body"].str[:2000]

df["body"] = df["body"].apply(clean_text)

# date time
df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

# sort by subreddit_id and time
df = df.sort_values(by=["subreddit_id", "created_utc"], ascending=[True, True])

# balance dataset
df = df.sample(frac=0.5, random_state=42)

# save test dataset csv
df.to_csv("reddit_clean_data.csv", index=False)

# --- Reshape: group comments by post (link_id) ---
grouped = (
    df.groupby(["link_id", "subreddit", "subreddit_id"])
    .apply(lambda g: pd.Series({
        "post_author": str(g["author"].iloc[0]),
        "post_body": str(g["body"].iloc[0]),
        "post_created_utc": g["created_utc"].min(),

        "comments": g.iloc[1:][["author", "body", "created_utc"]].rename(columns={
            "author": "comment_author",
            "body": "comment_body",
            "created_utc": "comment_utc"
        }).to_dict(orient="records")
    }))
    .reset_index()
)

grouped = grouped.rename(columns={"link_id": "post_id"})
grouped = grouped[["subreddit", "subreddit_id", "post_id",
                   "post_author", "post_body", "post_created_utc", "comments"]]

# view
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(grouped.head(5))
print(f"\nGrouped Data Shape: {grouped.shape}")

grouped.to_parquet("reddit_grouped.parquet", index=False)
print("\nSaved reddit_grouped.parquet")

# Display full parquet file
df_check = pd.read_parquet("reddit_grouped.parquet")

# Save to readable text file
with open("reddit_grouped_check.txt", "w") as f:
    for i, row in df_check.iterrows():
        f.write(f"\n{'='*80}\n")
        f.write(f"ROW {i}\n")
        f.write(f"{'='*80}\n")
        f.write(f"subreddit:        {row['subreddit']}\n")
        f.write(f"subreddit_id:     {row['subreddit_id']}\n")
        f.write(f"post_id:          {row['post_id']}\n")
        f.write(f"post_author:      {row['post_author']}\n")
        f.write(f"Post_body:        {row['post_body']}\n")
        f.write(f"post_created_utc: {row['post_created_utc']}\n")

        if len(row['comments']) > 0:
            f.write(f"\nCOMMENTS ({len(row['comments'])}):\n")

        for j, c in enumerate(row['comments']):
            f.write(f"  [{j}] author: {c['comment_author']}\n")
            f.write(f"      utc:    {c['comment_utc']}\n")
            f.write(f"      body:   {c['comment_body']}\n\n")


print("Saved as text to view")
