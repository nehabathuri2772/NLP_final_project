import joblib
import pandas as pd
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader

from fine_tuning_pipeline.reward import AggregatorModel
from utils import create_preference_pairs, PreferenceDataset, ranking_loss, RegressionDataset
from sklearn.preprocessing import StandardScaler

from constants import *

# Used to normalize
scaler = StandardScaler()

def prepare_dataset():
    print("Loading RL dataset...")
    df = pd.read_json(DETOX_OUTPUT_LOG_FILE, lines=True)

    print(f"Requested {REWARD_TRAINING_STRATEGY} Reward Model Training Strategy...")
    if REWARD_TRAINING_STRATEGY in ("PTSD", "pareto-strict", "soft-dominance"):
        # 1. Observe specific metrics
        X = df[REWARD_MODEL_METRICS.keys()].to_numpy(dtype=np.float32)
        X = scaler.fit_transform(X)

        pref_list = [pref == 1 for pref in REWARD_MODEL_METRICS.values()]
        pairs = create_preference_pairs(X, pref_list, method=REWARD_TRAINING_STRATEGY, max_pairs=MAX_PARETO_PAIRS)

        pairs_df = pd.DataFrame([
            {
                "vector_a": vec_a.tolist(),
                "vector_b": vec_b.tolist(),
                "preference": pref
            }
            for vec_a, vec_b, pref in pairs
        ])
        print(f"Saving Preference Pairs to {PREFERENCE_OUTPUT_FILE}")
        pairs_df.to_json(PREFERENCE_OUTPUT_FILE, orient="records", lines=True)
        return
    elif REWARD_TRAINING_STRATEGY == "judge":
        # Normalize and obtain score from LLM judge overall grade
        df["score"] = scaler.fit_transform(df[["llm_overall"]])

    elif REWARD_TRAINING_STRATEGY == "heuristic":
        # Run heuristic function for all examples and save as score
        # TODO: Implement heuristic function for RL
        heuristics = ...
        df["score"] = scaler.fit_transform(heuristics)
    elif REWARD_TRAINING_STRATEGY == "sft_judge":
        # Save only best examples for judge
        df = df[df["llm_overall"] >= JUDGE_THRESHOLD]
    elif REWARD_TRAINING_STRATEGY == "sft_heuristic":
        # Save only best examples from heuristic function
        # TODO: Implement heuristic function for RL
        heuristics = ...
        df["heuristic"] = heuristics
        df = df[df["heuristic"] >= HEURISTIC_THRESHOLD]
    else:
        raise ValueError(f"Unknown method: {REWARD_TRAINING_STRATEGY}")

    output_df = pd.DataFrame()
    # Check if using RL or SFT
    if "score" in df.columns:
        output_df["score"] = df["score"]

        metric_order = list(REWARD_MODEL_METRICS.keys())
        output_df["vector"] = df[metric_order].values.tolist()

        print(f"Saving Scored Outputs to {SCORED_OUTPUT_FILE}")
        output_df.to_json(SCORED_OUTPUT_FILE, orient="records", lines=True)
    else:
        output_df = df[["completion", "raw_response"]]
        print(f"Saving SFT Data to {SFT_OUTPUT_FILE}")
        output_df.to_json(SFT_OUTPUT_FILE, orient="records", lines=True, force_ascii=False)

def train_ranker(input_file=PREFERENCE_OUTPUT_FILE, output_dir=OUTPUT_CHECKPOINT_PATH, epochs=20, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PreferenceDataset(input_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    input_dim = len(REWARD_MODEL_METRICS.keys())
    model = AggregatorModel(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            for vec_a, vec_b, pref in batch:
                vec_a = vec_a.to(device)
                vec_b = vec_b.to(device)
                batch_loss += ranking_loss(model, vec_a, vec_b, pref)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Ranker Epoch {epoch+1}: avg loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), f"{output_dir}/ranker_model.pth")
    joblib.dump(scaler, f"{output_dir}/ranker_scaler.pkl")
    print(f"Ranker saved to {output_dir}")

def train_regressor(input_file=SCORED_OUTPUT_FILE, output_dir=OUTPUT_CHECKPOINT_PATH, epochs=100, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = RegressionDataset(input_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = len(REWARD_MODEL_METRICS.keys())
    model = AggregatorModel(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for vec, score in loader:
            vec, score = vec.to(device), score.to(device)
            pred = model(vec)
            loss = loss_fn(pred, score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Regressor Epoch {epoch+1}: MSE = {avg_loss:.6f}")

    torch.save(model.state_dict(), f"{output_dir}/regressor_model.pth")
    joblib.dump(scaler, f"{output_dir}/regressor_scaler.pkl")
    print(f"Regressor saved to {output_dir}")

if __name__ == "__main__":
    # 1. Prepare RL/SFT Dataset
    print("Preparing RL/SFT Dataset...")
    prepare_dataset()

    # 2. Decide which Reward Model to Train
    print("\nTraining Reward Model...")
    if REWARD_TRAINING_STRATEGY in ("PTSD", "pareto-strict", "soft-dominance"):
        # Pair-wise training
        train_ranker()
    elif REWARD_TRAINING_STRATEGY in ("judge", "heuristic"):
        # Direct score regression training
        train_regressor()
    else:
        print(f"No Reward Model Training Required -- SFT Training Chosen")


