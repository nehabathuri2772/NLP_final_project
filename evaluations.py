from typing import List

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify

from constants import EMBEDDING_MODEL

class DetoxEvaluator:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading DetoxEvaluator on {self.device}...")
        # Sentence embedding model for cosine similarity
        self.sim_model = SentenceTransformer(EMBEDDING_MODEL, device=str(self.device))

        # Toxicity classifier
        self.tox_model = Detoxify('original')

    def cosine_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        all_texts = texts1 + texts2
        # Embed texts together for less system use
        embeddings = self.sim_model.encode(all_texts, convert_to_tensor=True)
        n = len(texts1)
        emb1 = embeddings[:n]
        emb2 = embeddings[n:]

        similarities = F.cosine_similarity(emb1, emb2)
        return similarities.tolist()

    def toxicity_detection(self, texts: List[str]) -> List[dict]:
        out = self.tox_model.predict(texts)
        # TODO: Use more metrics than just toxicity
        results = []
        for i in range(len(texts)):
            results.append({
                "toxicity": float(out["toxicity"][i]),
                "severe_toxicity": float(out["severe_toxicity"][i])
            })
        return results

    def run_pipeline(self, original_texts: List[str], detoxified_texts: List[str]) -> List[dict]:
        # 1. Batch cosine similarity
        sims = self.cosine_similarity(original_texts, detoxified_texts)

        # 2. Batch toxicity detection
        tox_orig_list = self.toxicity_detection(original_texts)
        tox_new_list = self.toxicity_detection(detoxified_texts)

        # 3. Combine results
        results = []
        for i in range(len(original_texts)):
            results.append({
                "cosine_similarity": sims[i],
                "toxicity_change": tox_new_list[i]["toxicity"] - tox_orig_list[i]["toxicity"],
                "severe_toxicity_change": tox_new_list[i]["severe_toxicity"] - tox_orig_list[i]["severe_toxicity"],
            })
        return results