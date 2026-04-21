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

#Read only 5,000 rows — no memory issues
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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(5))

# balance dataset
df = df.sample(n=5000, random_state=42)


# # save test dataset
df.to_parquet("reddit_clean_data.parquet", index=False)
df.to_csv("reddit_clean_data.csv", index=False)

print()
print(f"Clean_data shape: {df.shape}")
print("Saved reddit_test_data.parquet and csv")