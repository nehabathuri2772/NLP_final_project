import random
from itertools import combinations

import torch


def pareto_preference_pair(vec_a: torch.Tensor, vec_b: torch.Tensor) -> int:
    """
    Returns:
        1 if a dominates b (a better or equal in all, strictly better in at least one)
       -1 if b dominates a
        0 if neither dominates (incomparable)
    """
    better_in_a = []
    better_in_b = []
    for i, (va, vb) in enumerate(zip(vec_a, vec_b)):
        if i == 1:  # similarity: higher is better
            a_better = va > vb
            b_better = vb > va
        else:  # toxicity, perplexity: lower is better
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

def soft_dominance(vec_a, vec_b):
    score = 0
    for i, (va, vb) in enumerate(zip(vec_a, vec_b)):
        if i == 1:
            if va > vb: score += 1
            elif vb > va: score -= 1
        else:
            if va < vb: score += 1
            elif vb < va: score -= 1
    return score


def create_preference_pairs(X, method="PTSD", max_pairs=20000):
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
            pref = pareto_preference_pair(vec_a, vec_b)
            if pref == 0:
                # Fallback to soft dominance
                score = soft_dominance(vec_a, vec_b)
                if score > 0:
                    pref = 1
                elif score < 0:
                    pref = -1
                else:
                    pref = 0
        elif method == "pareto_strict":
            pref = pareto_preference_pair(vec_a, vec_b)
        else:  # method == "soft"
            score = soft_dominance(vec_a, vec_b)
            if score > 0:
                pref = 1
            elif score < 0:
                pref = -1
            else:
                pref = 0

        if pref != 0:
            pairs.append((vec_a, vec_b, pref))

    return pairs