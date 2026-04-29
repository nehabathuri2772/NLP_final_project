import pandas as pd
import torch
import numpy as np

from utils import create_preference_pairs
from sklearn.preprocessing import StandardScaler

from constants import *

# Used to normalize
scaler = StandardScaler()

def aggregator_ranking_loss(aggregator, metric1, metric2, preference):
    """
    preference = 1 -> metric1 should get higher score than metric2
    """
    r1 = aggregator(metric1)
    r2 = aggregator(metric2)
    if preference == 1:
        logits = r1 - r2
    else:
        logits = r2 - r1
    return -torch.log(torch.sigmoid(logits) + 1e-8).mean()

def update_aggregator(aggregator, optimizer, pairs_batch):
    total_loss = torch.tensor(0.0, requires_grad=True)
    for m1, m2, pref in pairs_batch:
        loss = aggregator_ranking_loss(aggregator, m1.unsqueeze(0), m2.unsqueeze(0), pref)
        total_loss = total_loss + loss
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return total_loss.item()

if __name__ == "__main__":
    print("Loading RL dataset...")
    df = pd.read_json(DETOX_OUTPUT_LOG_FILE, lines=True)

    print(f"Requested {REWARD_TRAINING_STRATEGY} Reward Model Training Strategy...")
    if REWARD_TRAINING_STRATEGY in ("PTSD", "pareto-strict", "soft-dominance"):
        # 1. Observe specific metrics
        X = df[PARETO_METRICS.keys()].to_numpy(dtype=np.float32)
        pref_list = [pref == 1 for pref in PARETO_METRICS.values()]
        pairs = create_preference_pairs(X, pref_list, method=REWARD_TRAINING_STRATEGY, max_pairs=MAX_PARETO_PAIRS)

        pairs_df = pd.DataFrame([
            {
                "vec_a": vec_a.tolist(),
                "vec_b": vec_b.tolist(),
                "preference": pref
            }
            for vec_a, vec_b, pref in pairs
        ])
        print(f"Saving Preference Pairs to {PREFERENCE_OUTPUT_FILE}")
        pairs_df.to_json(PREFERENCE_OUTPUT_FILE, orient="records", lines=True)
        exit(0)
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

    # Check if using RL or SFT
    if "score" in df.columns:
        REQUIRED_COLUMNS.append("score")
    # Condense df to only required columns
    df = df[REQUIRED_COLUMNS]

    # Save to new jsonl
    if "score" in REQUIRED_COLUMNS:
        print(f"Saving Scored Outputs to {SCORED_OUTPUT_FILE}")
        df.to_json(SCORED_OUTPUT_FILE, orient="records", lines=True)
    else:
        print(f"Saving SFT Data to {SFT_OUTPUT_FILE}")
        df.to_json(SFT_OUTPUT_FILE, orient="records", lines=True)

