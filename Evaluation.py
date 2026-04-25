import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CSV_PATH = "detoxified_comments.csv"

# ── Evaluator ─────────────────────────────────────────────────────────────────
class DetoxEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading similarity model...")
        self.sim_model = SentenceTransformer(EMBEDDING_MODEL, device=str(self.device))

        print("Loading toxicity model...")
        self.tox_model = Detoxify('original', device=self.device)

    def cosine_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.sim_model.encode(text1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(text2, convert_to_tensor=True)
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    def toxicity_scores(self, text: str) -> dict:
        out = self.tox_model.predict(text)
        return {
            "toxicity":        out["toxicity"],
            "severe_toxicity": out["severe_toxicity"],
            "obscene":         out["obscene"],
            "threat":          out["threat"],
            "insult":          out["insult"],
            "identity_attack": out["identity_attack"],
        }

    def evaluate(self, original: str, detoxified: str) -> dict:
        sim = self.cosine_similarity(original, detoxified)

        tox_before = self.toxicity_scores(original)
        tox_after  = self.toxicity_scores(detoxified)

        return {
            "cosine_similarity":          sim,
            "toxicity_before":            tox_before["toxicity"],
            "toxicity_after":             tox_after["toxicity"],
            "toxicity_change":            tox_after["toxicity"] - tox_before["toxicity"],
            "severe_toxicity_before":     tox_before["severe_toxicity"],
            "severe_toxicity_after":      tox_after["severe_toxicity"],
            "obscene_before":             tox_before["obscene"],
            "obscene_after":              tox_after["obscene"],
            "threat_before":              tox_before["threat"],
            "threat_after":               tox_after["threat"],
            "insult_before":              tox_before["insult"],
            "insult_after":               tox_after["insult"],
            "identity_attack_before":     tox_before["identity_attack"],
            "identity_attack_after":      tox_after["identity_attack"],
        }


# ── Load Data ─────────────────────────────────────────────────────────────────
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")

# Drop rows where either column is empty
df = df.dropna(subset=["Post_body", "detoxified_body"])
df = df[df["Post_body"].str.strip() != ""]
df = df[df["detoxified_body"].str.strip() != ""]
print(f"Rows after cleaning: {len(df)}")

# ── Run Evaluation ────────────────────────────────────────────────────────────
evaluator = DetoxEvaluator()

print("\nRunning evaluation...")
results = []
for i, row in df.iterrows():
    result = evaluator.evaluate(
        str(row["Post_body"]),
        str(row["detoxified_body"])
    )
    results.append(result)

    if (len(results)) % 100 == 0:
        print(f"  Evaluated {len(results)}/{len(df)} rows...")

# ── Merge & Save ──────────────────────────────────────────────────────────────
eval_df = pd.DataFrame(results, index=df.index)
df = pd.concat([df, eval_df], axis=1)
df.to_csv("detoxified_evaluated.csv", index=False)
print("\nSaved to detoxified_evaluated.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EVALUATION SUMMARY")
print("="*55)
print(f"Total rows evaluated:       {len(df)}")
print(f"Avg Cosine Similarity:      {df['cosine_similarity'].mean():.4f}  (1.0 = perfect meaning preservation)")
print(f"Avg Toxicity Before:        {df['toxicity_before'].mean():.4f}")
print(f"Avg Toxicity After:         {df['toxicity_after'].mean():.4f}")
print(f"Avg Toxicity Change:        {df['toxicity_change'].mean():.4f}  (negative = improved ✅)")
print(f"Rows toxicity reduced:      {(df['toxicity_change'] < 0).sum()}/{len(df)}")
print(f"Rows toxicity increased:    {(df['toxicity_change'] > 0).sum()}/{len(df)}")
print(f"Avg Obscene reduction:      {(df['obscene_before'] - df['obscene_after']).mean():.4f}")
print(f"Avg Insult reduction:       {(df['insult_before'] - df['insult_after']).mean():.4f}")
print(f"Avg Identity attack reduction: {(df['identity_attack_before'] - df['identity_attack_after']).mean():.4f}")
print("="*55)