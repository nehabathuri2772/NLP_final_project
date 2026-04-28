import glob
import os

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from datasets import load_dataset
import re

from constants import CLEANED_PARQUET_PATH, CONDENSED_PARQUET_PATH, BATCH_SIZE, REDDIT_DATASET, LOCAL_DATASET_PATH

def load_reddit_data(local=False, local_path=LOCAL_DATASET_PATH):
    if local:
        print(f"Loading dataset from local folder: {local_path}")
        # Check for any parquet files locally
        parquet_files = glob.glob(os.path.join(local_path, "*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {local_path}")

        dataset = ds.dataset(local_path, format="parquet")

        # Return a generator that yields batches as DataFrames
        def batch_generator(batch_size=BATCH_SIZE):
            scanner = dataset.scanner(batch_size=batch_size)
            for batch in scanner.to_batches():
                yield batch.to_pandas()

        return batch_generator

    else:
        print(f"Loading dataset from Hugging Face hub: {REDDIT_DATASET}")
        hf_dataset = load_dataset(REDDIT_DATASET, split="train", streaming=True)

        # Return a generator that yields batches as DataFrames
        def batch_generator(batch_size=BATCH_SIZE):
            for batch_dict in hf_dataset.iter(batch_size=batch_size):
                yield pd.DataFrame(batch_dict)

        return batch_generator


def clean_text(text):
    # Removes URL and special characters to clean text
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_comment(comment):
    # Keep non‑empty, non‑deleted comments with length >= 20
    if comment.get("body") is None:
        return False
    body = comment["body"]
    if body in ("[deleted]", "[removed]"):
        return False
    if len(body) < 20:
        return False
    return True


def clean_comment(comment):
    # Keep only the columns needed for comment
    return {
        "link_id": comment["link_id"],
        "subreddit": comment["subreddit"],
        "subreddit_id": comment["subreddit_id"],
        "author": comment["author"],
        "body": clean_text(comment["body"]),
        "created_utc": comment["created_utc"],
    }


def stream_and_write_cleaned(dataset, output_path=CONDENSED_PARQUET_PATH, batch_size=BATCH_SIZE, num_batches=10):
    writer = None

    for batch_num,df_batch in enumerate(dataset(batch_size=batch_size)):
        # Clean all valid comments in this DataFrame
        cleaned_batch = []
        for _, row in df_batch.iterrows():
            comment = row.to_dict()
            # Validation and cleaning in stream
            if is_valid_comment(comment):
                cleaned_batch.append(clean_comment(comment))

        # Write the cleaned batch to Parquet
        if cleaned_batch:
            df_out = pd.DataFrame(cleaned_batch)
            table = pa.Table.from_pandas(df_out)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            print(f"Written batch {batch_num} ({len(cleaned_batch)} comments)")

        # Stop if reached limit
        if batch_num >= num_batches - 1:
            break

    if writer:
        writer.close()
    print(f"Finished. Saved output to {output_path}")


def group_comments_with_duckdb(input_path=CONDENSED_PARQUET_PATH, output_path=CLEANED_PARQUET_PATH):
    # Using duckdb to handle big data grouping
    con = duckdb.connect()

    # Register the Parquet file as a view
    con.execute(f"CREATE OR REPLACE VIEW comments AS SELECT * FROM '{input_path}'")

    # SQL query replicating the Spark logic
    query = """
    WITH ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY link_id ORDER BY created_utc) AS rn
        FROM comments
    ),
    post_data AS (
        SELECT link_id,
               subreddit,
               subreddit_id,
               author AS post_author,
               body AS post_body,
               created_utc AS post_created_utc
        FROM ranked
        WHERE rn = 1
    ),
    comments_list AS (
        SELECT link_id,
               LIST({
                   'comment_author': author,
                   'comment_body': body,
                   'comment_utc': created_utc
               }) AS comments
        FROM ranked
        WHERE rn > 1
        GROUP BY link_id
    )
    SELECT p.subreddit,
           p.subreddit_id,
           p.link_id AS post_id,
           p.post_author,
           p.post_body,
           p.post_created_utc,
           COALESCE(c.comments, []) AS comments
    FROM post_data p
    LEFT JOIN comments_list c ON p.link_id = c.link_id
    ORDER BY p.subreddit_id, p.post_created_utc
    """

    # Execute and write to Parquet (DuckDB streams via RDD writing method)
    con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT 'parquet')")
    print(f"Grouped data saved to {output_path}")

    # Preview first 5 rows
    sample = con.execute(f"{query} LIMIT 5").df()
    print("Sample output:")
    print(sample)

    # Remove old db parquet
    os.remove(input_path)
    print(f"Removed input file: {input_path}")

    con.close()

def run_cleaning_pipeline(local=True, num_batches=1):
    # Load data
    dataset = load_reddit_data(local=local)

    # Store files from dataset
    stream_and_write_cleaned(dataset, num_batches=num_batches)

    # Group files for cleaning
    group_comments_with_duckdb()

if __name__ == "__main__":
    run_cleaning_pipeline()