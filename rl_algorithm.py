import os

import torch
import torch.optim as optim
import numpy as np
from trl.experimental.ppo import PPOConfig, PPOTrainer
from datasets import Dataset as HFDataset

# Your existing modules
from evaluations import DetoxEvaluator
from model import DetoxificationModel, AggregatorModel

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

def train_using_pareto_ppo(batch_size: int = 4, num_epochs: int = 3, output_dir: str = OUTPUT_CHECKPOINT_PATH):
    """
    Train the detoxification model using Pareto‑driven Aggregator + PPO.
    Toxic prompts are loaded from the pre‑filtered Parquet file
    """
    # 1. Load toxic prompts
    toxic_prompts = df_toxic["toxic_comments"].tolist()
    print(f"Loaded {len(toxic_prompts)} toxic prompts for training.")

    # 2. Load detox model
    model = DetoxificationModel()

    # 3. Build formatted prompts using the model prompt builder
    formatted_prompts = []
    for toxic_text in toxic_prompts:
        # Use the same method that detoxify() uses
        formatted_prompt = model.build_prompt(toxic_text)
        formatted_prompts.append(formatted_prompt)

    # 4. Prepare dataset for PPO
    hf_dataset = HFDataset.from_dict({
        "prompt": formatted_prompts,
        "raw_toxic": toxic_prompts,
    })

    # 5. PPO configuration
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=batch_size,
        mini_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=1
    )
    trainer = PPOTrainer(
        args=ppo_config,
        model=model.model,
        processing_class=model.tokenizer,
        train_dataset=hf_dataset,
        ref_model=None,
    )

    # 6. Initialise Aggregator reward model
    aggregator = AggregatorModel(input_dim=3)
    agg_optimizer = optim.Adam(aggregator.parameters(), lr=1e-3)

    # 7. Generation parameters
    gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "pad_token_id": model.tokenizer.eos_token_id,
        "eos_token_id": model.tokenizer.eos_token_id,
    }

    # 8. Evaluation helper
    evaluator = DetoxEvaluator()

    # 9. Training loop
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(trainer.dataloader):
            prompts_batch = batch["prompt"]

            # --- Generate two responses per prompt ---
            response_tensors1 = trainer.generate(prompts_batch, **gen_kwargs)
            responses1 = [model.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors1]

            response_tensors2 = trainer.generate(prompts_batch, **gen_kwargs)
            responses2 = [model.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors2]

            # --- Compute metric vectors for both responses ---
            metric_vectors1 = []
            metric_vectors2 = []
            for i, prompt in enumerate(prompts_batch):
                eval1 = evaluator.run_pipeline(prompt, responses1[i])
                eval2 = evaluator.run_pipeline(prompt, responses2[i])

                # Normalise perplexity (assume max ~100, lower better)
                ppl_norm1 = max(0, min(1, eval1.get("perplexity", 100) / 100))
                ppl_norm2 = max(0, min(1, eval2.get("perplexity", 100) / 100))

                vec1 = torch.tensor([eval1.get("toxicity", 0.5),
                                     eval1.get("similarity", 0.5),
                                     ppl_norm1], dtype=torch.float32)
                vec2 = torch.tensor([eval2.get("toxicity", 0.5),
                                     eval2.get("similarity", 0.5),
                                     ppl_norm2], dtype=torch.float32)
                metric_vectors1.append(vec1)
                metric_vectors2.append(vec2)

            # --- Reward from aggregator for the first response (PPO uses this) ---
            with torch.no_grad():
                rewards = [aggregator(vec).item() for vec in metric_vectors1]

            # --- PPO update (improves the policy) ---
            trainer.step(prompts_batch, response_tensors1, rewards)

            # --- Build Pareto preference pairs to train the aggregator ---
            pairs = []
            for v1, v2 in zip(metric_vectors1, metric_vectors2):
                pref = pareto_preference_pair(v1, v2)
                if pref != 0:
                    pairs.append((v1.detach(), v2.detach(), pref))
                else:
                    soft = soft_dominance(v1, v2)
                    if soft > 0:
                        pairs.append((v1.detach(), v2.detach(), 1))
                    elif soft < 0:
                        pairs.append((v1.detach(), v2.detach(), -1))
            if pairs:
                loss_agg = update_aggregator(aggregator, agg_optimizer, pairs)

            # --- Logging ---
            if batch_idx % 20 == 0:
                avg_reward = np.mean(rewards)
                print(f"Epoch {epoch+1} | Batch {batch_idx:4d} | Avg reward: {avg_reward:.3f} | Pairs: {len(pairs)}")

        # End of epoch – save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_pretrained(os.path.join(output_dir, f"epoch_{epoch+1}"))
        torch.save(aggregator.state_dict(), os.path.join(output_dir, f"aggregator_epoch_{epoch+1}.pt"))

    # Final save
    trainer.save_pretrained(output_dir)
    torch.save(aggregator.state_dict(), os.path.join(output_dir, "aggregator_final.pt"))
    print(f"Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    train_using_pareto_ppo()