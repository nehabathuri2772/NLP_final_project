import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify

# These should be consistent eval metrics
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class DetoxEvaluator:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Sentence embedding model for cosine similarity
        self.sim_model = SentenceTransformer(EMBEDDING_MODEL, device=str(self.device))

        # Toxicity classifier
        self.tox_model = Detoxify('original')

    def cosine_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.sim_model.encode(text1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(text2, convert_to_tensor=True)
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    def toxicity_detection(self, text: str) -> dict:
        out = self.tox_model.predict(text)
        # TODO: Use more metrics than just toxicity
        return {
            "toxicity": out["toxicity"],
            "severe_toxicity": out["severe_toxicity"]
        }

    def run_pipeline(self, original_text: str, detoxified_text: str) -> dict:
        # Run each evaluation and return as an eval dict
        sim = self.cosine_similarity(original_text, detoxified_text)
        tox_orig = self.toxicity_detection(original_text)
        tox_new = self.toxicity_detection(detoxified_text)

        return {
            "cosine_similarity": sim,
            "toxicity_change": tox_new["toxicity"]-tox_orig["toxicity"],
            "severe_toxicity_change": tox_new["severe_toxicity"]-tox_orig["severe_toxicity"],
        }