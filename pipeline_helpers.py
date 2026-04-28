import json

import pandas as pd

from constants import *
from evaluations import DetoxEvaluator
import pyarrow.parquet as pq

from model import DetoxificationModel

def load_toxic_comments(input_path=CLEANED_PARQUET_PATH, output_path=LABELED_PARQUET_PATH, toxicity_threshold=0.5):
    """
    Reads a Parquet file containing comment chains, adds a 'toxic' boolean label
    to each comment based on toxicity score, and saves the full labeled dataset.
    """
    print(f"Loading dataset from: {input_path}")

    evaluator = DetoxEvaluator()
    modified_chunks = []

    # Read Parquet in chunks using pyarrow's iter_batches
    parquet_file = pq.ParquetFile(input_path)
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=PARQUET_LOAD_CHUNK_SIZE)):
        chunk = batch.to_pandas()

        # Iterate through each row in the chunk
        for idx, row in chunk.iterrows():
            comment_chain = row.get("comments", [])

            # Skip rows without a valid comment chain
            if len(comment_chain) == 0:
                continue

            # Update each comment in the chain by adding the 'toxic' label
            comment_bodies = [comment["comment_body"] for comment in comment_chain]
            all_scores = []

            # Process in batches
            for i in range(0, len(comment_bodies), DETOX_BATCH_SIZE):
                batch = comment_bodies[i:i + DETOX_BATCH_SIZE]
                print(f"\tRunning toxicity eval on {len(batch)} comments...")
                batch_scores = evaluator.toxicity_detection(batch)
                all_scores.extend(batch_scores)

            # Assign labels using scores collected
            for comment, score in zip(comment_chain, all_scores):
                comment['toxic'] = score["toxicity"] >= toxicity_threshold
                comment['toxicity_score'] = score["toxicity"]

        modified_chunks.append(chunk)
        print(f"Processed chunk {i+1} (batch size {len(chunk)})")

    # Concatenate all modified chunks into a single DataFrame
    df_labeled = pd.concat(modified_chunks, ignore_index=True)
    print(f"Total rows processed: {len(df_labeled)}")
    print(f"Total comments labeled: {df_labeled['comments'].apply(len).sum()}")

    # Save to a new Parquet file
    df_labeled.to_parquet(output_path, index=False)
    print(f"\nSuccessfully saved labeled dataset to: {output_path}")

    return df_labeled


def run_detox_evaluation(input_path=LABELED_PARQUET_PATH, output_path=DETOXIFIED_PARQUET_PATH, output_log_path=OUTPUT_LOG_FILE):
    evaluator = DetoxEvaluator()
    model = DetoxificationModel()

    parquet_file = pq.ParquetFile(input_path)
    modified_chunks = []

    with open(output_log_path, "a") as f:
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=PARQUET_LOAD_CHUNK_SIZE)):
            chunk = batch.to_pandas()

            for idx, row in chunk.iterrows():
                comment_chain = row.get("comments", [])
                if not len(comment_chain) > 0:
                    # Ignore empty comment chains
                    continue

                # Before detoxify, compute old chain avg
                old_avg_toxicity = sum(c["toxicity_score"] for c in comment_chain) / len(comment_chain)

                # Step 1: Detoxify all toxic comments in this chain
                toxic_results = []  # List: {comment, original, detoxified, response}
                for comment in comment_chain:
                    if comment.get("toxic", False):
                        toxic_text = comment.get("comment_body")
                        print(f"\t[RAW]: {toxic_text}")
                        response = model.detoxify(toxic_text)
                        print(f"\t[RES]: {response["completion"]}")
                        toxic_results.append({
                            "comment": comment,
                            "original": toxic_text,
                            "detoxified": response["completion"],
                            "response": response
                        })

                if not toxic_results:
                    # If entire chain non toxic, ignore
                    continue

                # Step 2: Bulk evaluation for all toxic comments in this chain
                originals = [r["original"] for r in toxic_results]
                detoxified = [r["detoxified"] for r in toxic_results]
                per_comment_metrics = evaluator.run_pipeline(originals, detoxified)

                # Step 3: Enrich each original comment dict with new fields
                for res, metrics in zip(toxic_results, per_comment_metrics):
                    comment = res["comment"]
                    comment["detoxified_comment_body"] = res["detoxified"]
                    comment["cosine_similarity"] = metrics["cosine_similarity"]
                    comment["toxicity_change"] = metrics["toxicity_change"]
                    comment["severe_toxicity_change"] = metrics["severe_toxicity_change"]
                    comment["toxicity_score"] = comment["toxicity_score"] + metrics["toxicity_change"]

                # Step 4: Write one JSON line per toxic comment (RL training format)
                for res, metrics in zip(toxic_results, per_comment_metrics):
                    json_row = {
                        "subreddit": row.get("subreddit"),
                        "subreddit_id": row.get("subreddit_id"),
                        "post_id": row.get("post_id"),
                        **res["response"],
                        **metrics
                    }
                    json_row.pop("prompt", None)
                    f.write(json.dumps(json_row) + "\n")

                # Collect toxicity scores for ALL comments in the chain
                n = len(per_comment_metrics)
                avg_cos = sum(m["cosine_similarity"] for m in per_comment_metrics) / n
                avg_tox_change = sum(m["toxicity_change"] for m in per_comment_metrics) / n
                avg_severe_change = sum(m["severe_toxicity_change"] for m in per_comment_metrics) / n

                new_avg_toxicity = sum(c["toxicity_score"] for c in comment_chain) / len(comment_chain)
                print(f"Metrics:\n\tCos-Avg: {avg_cos}\n\tDeltaToxAvg: {avg_tox_change}\n\tDeltaSevereAvg {avg_severe_change}")
                print(f"\tAvg Chain Toxicity {old_avg_toxicity} -> {new_avg_toxicity}\n")

                # Append statistics
                chunk.at[idx, "chain_cosine_similarity_avg"] = avg_cos
                chunk.at[idx, "chain_toxicity_change_avg"] = avg_tox_change
                chunk.at[idx, "chain_severe_toxicity_change_avg"] = avg_severe_change
                chunk.at[idx, "chain_toxic_comment_count"] = n
                chunk.at[idx, "chain_avg_old_toxicity"] = old_avg_toxicity
                chunk.at[idx, "chain_avg_new_toxicity"] = new_avg_toxicity

                f.flush()
                print(f"Comment chain {idx} completed: {len(toxic_results)}/{len(comment_chain)} detoxified")

            # After processing all rows in this chunk, append the enriched chunk
            modified_chunks.append(chunk)
            print(f"Finished batch {i + 1} (rows: {len(chunk)})")

    # Step 5: Save the full enriched Parquet file
    df_enriched = pd.concat(modified_chunks, ignore_index=True)
    df_enriched.to_parquet(output_path, index=False)
    print(f"\nSaved enriched Parquet file to: {output_path}")