from transformers import GenerationConfig

# Model Configs
GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=2048,
    do_sample=False, # No sampling since this is research work, needs consistent results
)
BATCH_SIZE = 10000
DETOX_BATCH_SIZE = 50
PARQUET_LOAD_CHUNK_SIZE = 10000

# Algorithm Hyperparameters
TOXICITY_THRESHOLD = 0.5

# Data loading
REDDIT_DATASET = "fddemarco/pushshift-reddit-comments"
LOCAL_DATASET_PATH = "./data/pushshift-reddit-comments"

# Parquet file paths for input/outputs
CONDENSED_PARQUET_PATH = "./data/cleaned_comments.parquet"
CLEANED_PARQUET_PATH = "./data/reddit_cleaned.parquet"
LABELED_PARQUET_PATH = "./data/reddit_cleaned_labeled.parquet"
DETOXIFIED_PARQUET_PATH = "./data/reddit_detoxified.parquet"

# Logging
OUTPUT_LOG_FILE = "./data/rl_data/detoxify_output.jsonl"
OUTPUT_CHECKPOINT_PATH = "./data/rl_data/checkpoints"