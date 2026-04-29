import json
import random
from itertools import combinations
from typing import Optional, List

import torch
from torch.utils.data import Dataset


def pareto_preference_pair(vec_a: torch.Tensor, vec_b: torch.Tensor, higher_is_better: List[bool]) -> int:
    """
    Compare two vectors using Pareto dominance.

    Args:
        vec_a, vec_b: 1D tensors of equal length
        higher_is_better: list of booleans with same length as vectors.
                          True  -> larger value is better for that dimension
                          False -> smaller value is better for that dimension
        Returns: 1 if a dominates b, -1 if b dominates a, 0 otherwise.
    """
    better_in_a = []
    better_in_b = []
    for i, (va, vb) in enumerate(zip(vec_a, vec_b)):
        if higher_is_better[i]:
            a_better = va > vb
            b_better = vb > va
        else:  # lower is better
            a_better = va < vb
            b_better = vb < va
        better_in_a.append(a_better)
        better_in_b.append(b_better)

    a_dominates = all(better_in_a) and any(better_in_a)
    b_dominates = all(better_in_b) and any(better_in_b)
    if a_dominates:
        return 1
    elif b_dominates:
        return -1
    else:
        return 0

def soft_dominance(vec_a: torch.Tensor, vec_b: torch.Tensor, higher_is_better: List[bool]) -> int:
    score = 0
    for i, (va, vb) in enumerate(zip(vec_a, vec_b)):
        if higher_is_better[i]:
            if va > vb:
                score += 1
            elif vb > va:
                score -= 1
        else:
            if va < vb:
                score += 1
            elif vb < va:
                score -= 1
    return score


def create_preference_pairs(X, pref_list: List[bool], method="PTSD", max_pairs=20000):
    """
    X: numpy array of normalised metric vectors
    method:
        "PTSD"          -> use Pareto Soft Dominance --> Uses pareto, fall back to soft dominance.
        "pareto_strict" -> only keep pairs with strict Pareto dominance
        "soft"          -> only use soft dominance
    Returns list of (tensor_a, tensor_b, pref) where pref = 1 if a is better, -1 if b is better.
    """
    pairs = []
    n = len(X)
    indices = list(range(n))
    # Sample pairs if too many
    if n * (n - 1) // 2 > max_pairs:
        sampled_pairs = set()
        while len(sampled_pairs) < max_pairs:
            i, j = random.sample(indices, 2)
            if i > j:
                i, j = j, i
            sampled_pairs.add((i, j))
        pair_indices = list(sampled_pairs)
    else:
        pair_indices = list(combinations(indices, 2))

    for i, j in pair_indices:
        vec_a = torch.tensor(X[i], dtype=torch.float32)
        vec_b = torch.tensor(X[j], dtype=torch.float32)

        if method == "pareto":
            # Try strict Pareto dominance first
            pref = pareto_preference_pair(vec_a, vec_b, pref_list)
            if pref == 0:
                # Fallback to soft dominance
                score = soft_dominance(vec_a, vec_b, pref_list)
                if score > 0:
                    pref = 1
                elif score < 0:
                    pref = -1
                else:
                    pref = 0
        elif method == "pareto_strict":
            pref = pareto_preference_pair(vec_a, vec_b, pref_list)
        else:  # method == "soft"
            score = soft_dominance(vec_a, vec_b, pref_list)
            if score > 0:
                pref = 1
            elif score < 0:
                pref = -1
            else:
                pref = 0

        if pref != 0:
            pairs.append((vec_a, vec_b, pref))

    return pairs

class RegressionDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                vec = torch.tensor(item['vector'], dtype=torch.float32)
                score = torch.tensor(item['score'], dtype=torch.float32)
                self.data.append((vec, score))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class PreferenceDataset(Dataset):
    def __init__(self, jsonl_path):
        self.pairs = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                vec_a = torch.tensor(data['vector_a'], dtype=torch.float32)
                vec_b = torch.tensor(data['vector_b'], dtype=torch.float32)
                pref = data['preference']
                self.pairs.append((vec_a, vec_b, pref))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def ranking_loss(model, vec_a, vec_b, pref):
    score_a = model(vec_a.unsqueeze(0))
    score_b = model(vec_b.unsqueeze(0))
    if pref == 1:
        logits = score_a - score_b
    else:
        logits = score_b - score_a
    return -torch.log(torch.sigmoid(logits) + 1e-8).mean()