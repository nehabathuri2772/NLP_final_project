import json
import os

from data_cleaning import run_cleaning_pipeline
from evaluations import DetoxEvaluator
from model import DetoxificationModel

import pyarrow.dataset as ds

DATA_PATH = "./data/reddit_cleaned.parquet"
OUTPUT_LOG_PATH = "./data/rl_data/detoxify_output.jsonl"
BATCH_SIZE = 10000
TOXICITY_THRESHOLD = 0.5

def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main():
    # 1) Check if cleaned data exists
    if not os.path.exists(DATA_PATH):
        print("Cleaned data not found. Running cleaning pipleine...")
        run_cleaning_pipeline()
    else:
        print(f"Found reddit cleaned data at: {DATA_PATH}.")

    # 2) Initialize evaluator + model
    evaluator = DetoxEvaluator()
    model = DetoxificationModel()

    # 3) Load dataset using pyarrow
    print("Initializing dataset scanner...")
    dataset = ds.dataset(DATA_PATH, format="parquet")

    scanner = ds.Scanner.from_dataset(
        dataset,
        columns=["subreddit", "subreddit_id", "post_id", "comments"],
        batch_size=BATCH_SIZE
    )

    # Variables to track
    total_comments = 0
    toxic_comments = 0

    # 4) Stream comment chains in batches
    with open(OUTPUT_LOG_PATH, "a") as f:
        for record_batch in scanner.to_batches():
            records = record_batch.to_pylist()

            # Extract comments from the batch
            for row in records:
                comment_chain = row["comments"]
                if not comment_chain:
                    # Ignore posts without comments
                    continue

                for comment in comment_chain:
                    toxic_text = comment.get("comment_body") # Toxic text
                    total_comments += 1

                    # 5) Evaluate toxicity of comment body initially (low toxicity = ignore)
                    score = evaluator.toxicity_detection(toxic_text)

                    # 6) Ignore if not toxic enough
                    if score["toxicity"] < TOXICITY_THRESHOLD:
                        continue

                    # 7) Run detoxify model if message is toxic
                    toxic_comments += 1
                    response = model.detoxify(toxic_text)

                    # 8) Run full evaluations
                    evaluations = evaluator.run_pipeline(toxic_text, response["completion"])

                    # 9) Store full pipeline results as a row
                    toxicity_data_row = {
                        "subreddit": row["subreddit"],
                        "subreddit_id": row["subreddit_id"],
                        "post_id": row["post_id"],
                        **response, **evaluations
                    }

                    toxicity_data_row.pop("prompt") # Remove prompt column since we do not need it
                    f.write(json.dumps(toxicity_data_row) + "\n")

                # Every time comment chain is completed, push to file
                f.flush()



if __name__ == "__main__":
    main()
