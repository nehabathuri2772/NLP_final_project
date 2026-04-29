from transformers import GenerationConfig

# Model Configs
GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
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

# File paths for input/outputs
CONDENSED_PARQUET_PATH = "./data/cleaned_comments.parquet"
CLEANED_PARQUET_PATH = "./data/reddit_cleaned.parquet"
LABELED_PARQUET_PATH = "./data/reddit_cleaned_labeled.parquet"
DETOXIFIED_PARQUET_PATH = "./data/reddit_detoxified.parquet"

DETOX_OUTPUT_LOG_FILE = "./data/training_data/detoxify_output.jsonl"
SCORED_OUTPUT_FILE = "./data/training_data/outputs_scored.jsonl"
SFT_OUTPUT_FILE = "./data/training_data/sft_outputs.jsonl"
PREFERENCE_OUTPUT_FILE = "./data/training_data/reward_preferences.jsonl"

OUTPUT_CHECKPOINT_PATH = "./data/training_data/checkpoints"

# Reward Model Training
REWARD_TRAINING_STRATEGY = "PTSD"
# 1 means higher is better, -1 means lower is better
PARETO_METRICS = {
    "toxicity_change": -1,
    "cosine_similarity": 1,
    "length_ratio": 1,
    "rougeL": 1,
    "llm_meaning_preservation": 1,
    "llm_fluency": 1,
    "llm_overall": 1,
}
MAX_PARETO_PAIRS = 20000

HEURISTIC_THRESHOLD = 0.5
JUDGE_THRESHOLD = 5
REQUIRED_COLUMNS = ["completion", "raw_response"]