import joblib
import numpy as np
import pandas as pd
from datasets import Dataset
from trl import GRPOTrainer

from constants import *
from evaluation_pipeline.evaluations import DetoxEvaluator
from fine_tuning_pipeline.reward import AggregatorModel
from model import DetoxificationModel
from peft import get_peft_model

REWARD_MODEL_TYPE = "ranker"
REWARD_MODEL_PATH = f"{OUTPUT_CHECKPOINT_PATH}/{REWARD_MODEL_TYPE}_model.pth"
SCALER_MODEL_PATH = f"{OUTPUT_CHECKPOINT_PATH}/{REWARD_MODEL_TYPE}_scaler.pkl"

class GRPOTrainingPipeline:

    def __init__(self):
        print("Initializing GRPO Trainer...")
        self.model, self.reward_model, self.scaler, self.dataset, self.evaluator = self.load_pipeline()

    def load_pipeline(self):
        print("Loading policy model...")
        model = DetoxificationModel()

        print("Loading reward model...")
        input_dim = len(REWARD_MODEL_METRICS.keys())
        reward_model = AggregatorModel(input_dim).to(DEVICE)
        reward_model.load_state_dict(torch.load(REWARD_MODEL_PATH, map_location=DEVICE))
        reward_model.eval()

        for param in reward_model.parameters():
            param.requires_grad = False

        print("Loading dataset...")
        df_prompts = pd.read_json(DETOX_OUTPUT_LOG_FILE, lines=True)
        df_prompts['prompt'] = df_prompts.apply(
            lambda row: row['raw_response'].replace(row['completion'], ''),
            axis=1
        )

        # Keep only the text column
        dataset = Dataset.from_pandas(df_prompts[['prompt']])

        print("Loading Scaler...")
        scaler = joblib.load(SCALER_MODEL_PATH)

        print("Loading Evaluator...")
        evaluator = DetoxEvaluator()

        return model, reward_model, scaler, dataset, evaluator

    def compute_metrics(self, original_texts: list, detoxified_texts: list):
        """Return normalised metric vector."""
        evals = self.evaluator.run_pipeline(original_texts, detoxified_texts)
        raw = np.array([
            [e[metric] if e[metric] is not None else 0.0 for metric in REWARD_MODEL_METRICS.keys()]
            for e in evals
        ], dtype=np.float32)
        norm = self.scaler.transform(raw) # shape (batch_size, n_features)
        return norm

    def get_reward(self, original_texts: list, detoxified_texts: list):
        """Reward from the metric reward model."""
        metrics_vec = self.compute_metrics(original_texts, detoxified_texts)
        vec_tensor = torch.tensor(metrics_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            rewards = self.reward_model(vec_tensor) # shape (batch_size, 1)
        return rewards.cpu().tolist()

    # GRPOTrainer wants [list prompts, list completions] -> list floats score.
    def reward_func(self, prompts, completions, **kwargs) -> list:
        """
        prompts: list of original prompts (note: this includes system prompt etc)
        completions: list of generated detoxified texts
        Returns: list of rewards
        """
        # Retrieve toxic_texts from prompts since data was not organized well
        toxic_texts = [text.split("text: '")[1].split("'\n")[0] for text in prompts]

        rewards = self.get_reward(toxic_texts, completions)
        return rewards

def run_grpo_training():
    pipeline = GRPOTrainingPipeline()

    print("Converting to PEFT Model...")
    # Replacing so garbage collector removes old model
    pipeline.model.model = get_peft_model(pipeline.model.model, LORA_CONFIG)

    print("Training...")
    trainer = GRPOTrainer(
        model=pipeline.model.model,
        train_dataset=pipeline.dataset,
        reward_funcs=[pipeline.reward_func],
        args=GRPO_CONFIG,
        processing_class=pipeline.model.tokenizer,
    )
    trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_CHECKPOINT_PATH)
    print(f"Final policy model saved to {OUTPUT_CHECKPOINT_PATH}")

if __name__ == "__main__":
    run_grpo_training()