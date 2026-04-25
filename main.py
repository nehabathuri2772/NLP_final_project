import os

from data_cleaning import run_cleaning_pipeline
from evaluations import DetoxEvaluator
from model import DetoxificationModel

import pyarrow.dataset as ds

DATA_PATH = "./data/reddit_cleaned.parquet"
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
        columns=["comments"],
        batch_size=BATCH_SIZE
    )

    # Variables to track
    total_comments = 0
    toxic_comments = 0

    # 4) Stream comment chains in batches
    for record_batch in scanner.to_batches():
        comments_column = record_batch.column("comments")

        # Extract comments from the batch
        for comment_chain in comments_column.to_pylist():
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
                toxicity_data_row = {"toxic_text": toxic_text, **response, **evaluations}
                # TODO: Append data to store externally



if __name__ == "__main__":
    main()
