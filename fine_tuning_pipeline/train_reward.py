import os

import pandas as pd
import torch
import torch.optim as optim
import numpy as np
from trl.experimental.ppo import PPOConfig, PPOTrainer
from datasets import Dataset as HFDataset

from evaluation_pipeline.evaluations import DetoxEvaluator
from utils import create_preference_pairs
from model import DetoxificationModel
from reward import AggregatorModel

def aggregator_ranking_loss(aggregator, metric1, metric2, preference):
    """preference = 1 -> metric1 should score higher"""
    r1 = aggregator(metric1)
    r2 = aggregator(metric2)
    if preference == 1:
        logits = r1 - r2
    else:
        logits = r2 - r1
    return -torch.log(torch.sigmoid(logits) + 1e-8).mean()

def update_aggregator(aggregator, optimizer, pairs_batch):
    """pairs_batch: list of (metric_a, metric_b, pref)"""
    total_loss = 0.0
    for m1, m2, pref in pairs_batch:
        loss = aggregator_ranking_loss(aggregator, m1, m2, pref)
        total_loss += loss.item()
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return total_loss

METHOD = "PTSD"
PARETO_METRICS = ["toxicity_change", "cosine_similarity", "length_ratio"]
if __name__ == "__main__":
    print("Loading RL dataset...")
    df = pd.read_json("filtered_best.jsonl", lines=True)

    if METHOD in ("PTSD", "pareto-strict", "soft-dominance"):
        # 1. Observe specific metrics
        X = df[PARETO_METRICS].to_numpy(dtype=np.float32)
        pairs = create_preference_pairs(X, method=METHOD)
    elif METHOD == "judge":
        # 1. Condense dataset
        pass
    elif METHOD == "heuristic":
        # Run heuristic function for all examples
        heuristics = ...