from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify

from constants import EMBEDDING_MODEL
from llm_judge import LLMJudge


class DetoxEvaluator:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading DetoxEvaluator on {self.device}...")

        print("\tLoading Similarity Embedding metric...")
        self.sim_model = SentenceTransformer(EMBEDDING_MODEL, device=self.device)

        print("\tLoading Detoxify metric...")
        self.tox_model = Detoxify('original', device=self.device)

        print("\tLoading ROUGE metric...")
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        self.llm_judge = LLMJudge()

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
        results = []
        for i in range(len(texts)):
            results.append({
                "toxicity": float(out["toxicity"][i]),
                "severe_toxicity": float(out["severe_toxicity"][i]),
                "obscene": float(out["obscene"][i]),
                "threat": float(out["threat"][i]),
                "insult": float(out["insult"][i]),
            })
        return results

    def length_ratio(self, original: List[str], detoxified: List[str]) -> List[float]:
        return [(len(d.split()) / len(o.split())) for o, d in zip(original, detoxified)]

    def bleu(self, original: List[str], detoxified: List[str]) -> List[float]:
        smooth_f = SmoothingFunction().method1

        return [
            sentence_bleu([o.lower().split()], d.lower().split(), smoothing_function=smooth_f)
            for o, d in zip(original, detoxified)
        ]

    def rouge_score(self, original: List[str], detoxified: List[str]) -> List[float]:
        return [
            self.rouge_scorer.score(o, d)["rougeL"].fmeasure
            for o, d in zip(original, detoxified)
        ]

    def llm_judge_batch(self, originals: List[str], detoxified: List[str]) -> List[Dict[str, Any]]:
        return [self.llm_judge.judge(o, d) for o, d in zip(originals, detoxified)]

    def run_pipeline(self, original_texts: List[str], detoxified_texts: List[str]) -> List[dict]:
        # 1. Batch cosine similarity
        sims = self.cosine_similarity(original_texts, detoxified_texts)

        # 2. Batch toxicity detection
        tox_orig_list = self.toxicity_detection(original_texts)
        tox_new_list = self.toxicity_detection(detoxified_texts)

        # 3. Batch length ratio
        length_ratios = self.length_ratio(original_texts, detoxified_texts)

        # 4. Batch BLEU score
        bleu_scores = self.bleu(original_texts, detoxified_texts)

        # 5. Batch ROUGE-L
        rouge_scores = self.rouge_score(original_texts, detoxified_texts)

        # 6. LLM judge metrics
        llm_judgments = self.llm_judge_batch(original_texts, detoxified_texts)

        # 7. Combine results
        results = []
        for i in range(len(original_texts)):
            j = llm_judgments[i]
            results.append({
                # Cosine similarity item
                "cosine_similarity": sims[i],

                # Detoxify items
                "toxicity_change": (tox_new_list[i]["toxicity"] - tox_orig_list[i]["toxicity"]),
                "severe_toxicity_change": (tox_new_list[i]["severe_toxicity"] - tox_orig_list[i]["severe_toxicity"]),

                # Length ratio item
                "length_ratio": length_ratios[i],

                # Other metrics
                "bleu": bleu_scores[i],
                "rougeL": rouge_scores[i],

                # LLM judge metrics
                "llm_toxicity_removal": j.get("toxicity_removal"),
                "llm_meaning_preservation": j.get("meaning_preservation"),
                "llm_fluency": j.get("fluency"),
                "llm_refusal": j.get("refusal"),
                "llm_overall": j.get("overall"),
                "llm_reasoning": j.get("reasoning"),
            })
        return results