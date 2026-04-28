import os

from constants import CLEANED_PARQUET_PATH, LABELED_PARQUET_PATH
from data_cleaning import run_cleaning_pipeline
from pipeline_helpers import load_toxic_comments, run_detox_evaluation

def main():
    # 1) Check if cleaned data exists
    if not os.path.exists(CLEANED_PARQUET_PATH):
        print("Cleaned data not found. Running cleaning pipleine...")
        run_cleaning_pipeline()
    else:
        print(f"Found reddit cleaned data at: {CLEANED_PARQUET_PATH}.")

    # 2) Load pre‑labeled dataset (adds 'toxic' key to each comment)
    if not os.path.exists(LABELED_PARQUET_PATH):
        print("Labeled toxic data not found. Running labelling pipeline...")
        load_toxic_comments()
    else:
        print(f"Found labeled data at: {LABELED_PARQUET_PATH}.")

    # 3) Initialize evaluator + model
    print("Running detoxification evaluation pipeline...")
    run_detox_evaluation()



if __name__ == "__main__":
    main()
