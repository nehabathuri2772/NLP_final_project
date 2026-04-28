# """import torch
# import torch.nn.functional as F
# import os
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from detoxify import Detoxify
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# # ── Config ────────────────────────────────────────────────────────────────────
#
# HF_TOKEN = "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"
# os.environ["HF_TOKEN"] = HF_TOKEN
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CSV_PATH = "detoxified_comments.csv"
#
# # ── Evaluator ─────────────────────────────────────────────────────────────────
# class DetoxEvaluator:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {self.device}")
#
#         print("Loading similarity model...")
#         self.sim_model = SentenceTransformer(EMBEDDING_MODEL, device=str(self.device))
#
#         print("Loading toxicity model...")
#         self.tox_model = Detoxify('original', device=self.device)
#
#     def cosine_similarity(self, text1: str, text2: str) -> float:
#         emb1 = self.sim_model.encode(text1, convert_to_tensor=True)
#         emb2 = self.sim_model.encode(text2, convert_to_tensor=True)
#         return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
#
#     def toxicity_scores(self, text: str) -> dict:
#         out = self.tox_model.predict(text)
#         return {
#             "toxicity":        out["toxicity"],
#             "severe_toxicity": out["severe_toxicity"],
#             "obscene":         out["obscene"],
#             "threat":          out["threat"],
#             "insult":          out["insult"],
#             "identity_attack": out["identity_attack"],
#         }
#
#     def evaluate(self, original: str, detoxified: str) -> dict:
#         sim = self.cosine_similarity(original, detoxified)
#
#         tox_before = self.toxicity_scores(original)
#         tox_after  = self.toxicity_scores(detoxified)
#
#         return {
#             "cosine_similarity":          sim,
#             "toxicity_before":            tox_before["toxicity"],
#             "toxicity_after":             tox_after["toxicity"],
#             "toxicity_change":            tox_after["toxicity"] - tox_before["toxicity"],
#             "severe_toxicity_before":     tox_before["severe_toxicity"],
#             "severe_toxicity_after":      tox_after["severe_toxicity"],
#             "obscene_before":             tox_before["obscene"],
#             "obscene_after":              tox_after["obscene"],
#             "threat_before":              tox_before["threat"],
#             "threat_after":               tox_after["threat"],
#             "insult_before":              tox_before["insult"],
#             "insult_after":               tox_after["insult"],
#             "identity_attack_before":     tox_before["identity_attack"],
#             "identity_attack_after":      tox_after["identity_attack"],
#         }
#
#
# # ── Load Data ─────────────────────────────────────────────────────────────────
# print(f"Loading data from {CSV_PATH}...")
# df = pd.read_csv(CSV_PATH)
# print(f"Loaded {len(df)} rows")
#
# # Drop rows where either column is empty
# df = df.dropna(subset=["Post_body", "detoxified_body"])
# df = df[df["Post_body"].str.strip() != ""]
# df = df[df["detoxified_body"].str.strip() != ""]
# print(f"Rows after cleaning: {len(df)}")
#
# # ── Run Evaluation ────────────────────────────────────────────────────────────
# evaluator = DetoxEvaluator()
#
# print("\nRunning evaluation...")
# results = []
# for i, row in df.iterrows():
#     result = evaluator.evaluate(
#         str(row["Post_body"]),
#         str(row["detoxified_body"])
#     )
#     results.append(result)
#
#     if (len(results)) % 100 == 0:
#         print(f"  Evaluated {len(results)}/{len(df)} rows...")
#
# # ── Merge & Save ──────────────────────────────────────────────────────────────
# eval_df = pd.DataFrame(results, index=df.index)
# df = pd.concat([df, eval_df], axis=1)
# df.to_csv("detoxified_evaluated.csv", index=False)
# print("\nSaved to detoxified_evaluated.csv")
#
# # ── Summary ───────────────────────────────────────────────────────────────────
# print("\n" + "="*55)
# print("EVALUATION SUMMARY")
# print("="*55)
# print(f"Total rows evaluated:       {len(df)}")
# print(f"Avg Cosine Similarity:      {df['cosine_similarity'].mean():.4f}  (1.0 = perfect meaning preservation)")
# print(f"Avg Toxicity Before:        {df['toxicity_before'].mean():.4f}")
# print(f"Avg Toxicity After:         {df['toxicity_after'].mean():.4f}")
# print(f"Avg Toxicity Change:        {df['toxicity_change'].mean():.4f}  (negative = improved ✅)")
# print(f"Rows toxicity reduced:      {(df['toxicity_change'] < 0).sum()}/{len(df)}")
# print(f"Rows toxicity increased:    {(df['toxicity_change'] > 0).sum()}/{len(df)}")
# print(f"Avg Obscene reduction:      {(df['obscene_before'] - df['obscene_after']).mean():.4f}")
# print(f"Avg Insult reduction:       {(df['insult_before'] - df['insult_after']).mean():.4f}")
# print(f"Avg Identity attack reduction: {(df['identity_attack_before'] - df['identity_attack_after']).mean():.4f}")
# print("="*55)"""
#
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# import os
# import torch
# import torch.nn.functional as F
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from detoxify import Detoxify
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer
# from bert_score import score as bert_score_fn
#
# HF_TOKEN = os.environ.get("HF_TOKEN")
#
# # ── Config ────────────────────────────────────────────────────────────────────
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# CSV_PATH = "detoxified_comments.csv"
#
# REFUSAL_PHRASES = [
#     "i can't help",
#     "i can't fulfill",
#     "i cannot assist",
#     "i can't assist",
#     "i cannot provide",
#     "i can't provide",
#     "i'm not able",
#     "i cannot help",
#     "i can't do",
#     "i won't",
# ]
#
# # ── Evaluator ─────────────────────────────────────────────────────────────────
# class DetoxEvaluator:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {self.device}")
#
#         print("Loading similarity model...")
#         self.sim_model = SentenceTransformer(
#             EMBEDDING_MODEL, device=str(self.device)
#         )
#
#         print("Loading toxicity model...")
#         self.tox_model = Detoxify('original', device=self.device)
#
#         print("Loading ROUGE scorer...")
#         self.rouge_scorer = rouge_scorer.RougeScorer(
#             ["rougeL"], use_stemmer=True
#         )
#
#     def cosine_similarity(self, text1: str, text2: str) -> float:
#         emb1 = self.sim_model.encode(text1, convert_to_tensor=True)
#         emb2 = self.sim_model.encode(text2, convert_to_tensor=True)
#         return F.cosine_similarity(
#             emb1.unsqueeze(0), emb2.unsqueeze(0)
#         ).item()
#
#     def toxicity_scores(self, text: str) -> dict:
#         out = self.tox_model.predict(text)
#         return {
#             "toxicity":        out["toxicity"],
#             "severe_toxicity": out["severe_toxicity"],
#             "obscene":         out["obscene"],
#             "threat":          out["threat"],
#             "insult":          out["insult"],
#             "identity_attack": out["identity_attack"],
#         }
#
#     def is_refusal(self, text: str) -> bool:
#         text_lower = text.lower().strip()
#         return any(phrase in text_lower for phrase in REFUSAL_PHRASES)
#
#     def length_ratio(self, original: str, detoxified: str) -> float:
#         orig_len = len(original.split())
#         if orig_len == 0:
#             return 0.0
#         return len(detoxified.split()) / orig_len
#
#     def bleu(self, original: str, detoxified: str) -> float:
#         reference = [original.lower().split()]
#         hypothesis = detoxified.lower().split()
#         smoothie = SmoothingFunction().method1
#         return sentence_bleu(
#             reference, hypothesis,
#             smoothing_function=smoothie
#         )
#
#     def rouge_score(self, original: str, detoxified: str) -> float:
#         scores = self.rouge_scorer.score(original, detoxified)
#         return scores["rougeL"].fmeasure
#
#     def evaluate(self, original: str, detoxified: str) -> dict:
#         sim   = self.cosine_similarity(original, detoxified)
#         tox_before = self.toxicity_scores(original)
#         tox_after  = self.toxicity_scores(detoxified)
#         refusal    = self.is_refusal(detoxified)
#         length_r   = self.length_ratio(original, detoxified)
#         bleu_s     = self.bleu(original, detoxified)
#         rouge_s    = self.rouge_score(original, detoxified)
#
#         return {
#             # Similarity
#             "cosine_similarity":          sim,
#             "bleu_score":                 bleu_s,
#             "rouge_l":                    rouge_s,
#             # Toxicity
#             "toxicity_before":            tox_before["toxicity"],
#             "toxicity_after":             tox_after["toxicity"],
#             "toxicity_change":            tox_after["toxicity"] - tox_before["toxicity"],
#             "severe_toxicity_before":     tox_before["severe_toxicity"],
#             "severe_toxicity_after":      tox_after["severe_toxicity"],
#             "obscene_before":             tox_before["obscene"],
#             "obscene_after":              tox_after["obscene"],
#             "threat_before":              tox_before["threat"],
#             "threat_after":               tox_after["threat"],
#             "insult_before":              tox_before["insult"],
#             "insult_after":               tox_after["insult"],
#             "identity_attack_before":     tox_before["identity_attack"],
#             "identity_attack_after":      tox_after["identity_attack"],
#             # Quality
#             "is_refusal":                 refusal,
#             "length_ratio":               length_r,
#         }
#
#
# # ── Load Data ─────────────────────────────────────────────────────────────────
# print(f"Loading data from {CSV_PATH}...")
# df = pd.read_csv(CSV_PATH)
# print(f"Loaded {len(df)} rows")
#
# df = df.dropna(subset=["Post_body", "detoxified_body"])
# df = df[df["Post_body"].str.strip() != ""]
# df = df[df["detoxified_body"].str.strip() != ""]
# print(f"Rows after cleaning: {len(df)}")
#
# # ── Run Row by Row Evaluation ─────────────────────────────────────────────────
# evaluator = DetoxEvaluator()
#
# print("\nRunning evaluation...")
# results = []
# for i, row in df.iterrows():
#     result = evaluator.evaluate(
#         str(row["Post_body"]),
#         str(row["detoxified_body"])
#     )
#     results.append(result)
#     if len(results) % 100 == 0:
#         print(f"  Evaluated {len(results)}/{len(df)} rows...")
#
# # ── BERTScore (run on full list at once — faster) ─────────────────────────────
# print("\nComputing BERTScore...")
# originals    = df["Post_body"].tolist()
# detoxifieds  = df["detoxified_body"].tolist()
#
# P, R, F1 = bert_score_fn(
#     detoxifieds,
#     originals,
#     lang="en",
#     verbose=True
# )
#
# # ── Merge & Save ──────────────────────────────────────────────────────────────
# eval_df = pd.DataFrame(results, index=df.index)
# eval_df["bert_score_f1"] = F1.numpy()
# df = pd.concat([df, eval_df], axis=1)
# df.to_csv("detoxified_evaluated.csv", index=False)
# print("\nSaved to detoxified_evaluated.csv")
#
# # ── Summary ───────────────────────────────────────────────────────────────────
# # ── Summary ───────────────────────────────────────────────────────────────────
# print("\n" + "="*55)
# print("EVALUATION SUMMARY")
# print("="*55)
# print(f"Total rows evaluated:          {len(df)}")
# print(f"Refusal rate:                  {df['is_refusal'].sum()}/{len(df)} ({df['is_refusal'].mean()*100:.1f}%)")
# print(f"Avg Length Ratio:              {df['length_ratio'].mean():.4f}  (1.0 = same length)")
# print(f"Avg Cosine Similarity:         {df['cosine_similarity'].mean():.4f}")
# print(f"Avg BERTScore F1:              {df['bert_score_f1'].mean():.4f}")
# print(f"Avg BLEU Score:                {df['bleu_score'].mean():.4f}")
# print(f"Avg ROUGE-L:                   {df['rouge_l'].mean():.4f}")
# print(f"Avg Toxicity Before:           {df['toxicity_before'].iloc[:, 0].mean():.4f}" if df['toxicity_before'].ndim > 1 else f"Avg Toxicity Before:           {df['toxicity_before'].mean():.4f}")
# print(f"Avg Toxicity After:            {df['toxicity_after'].mean():.4f}")
# print(f"Avg Toxicity Change:           {df['toxicity_change'].mean():.4f}")
# print(f"Rows toxicity reduced:         {(df['toxicity_change'] < 0).sum()}/{len(df)}")
# print(f"Rows toxicity increased:       {(df['toxicity_change'] > 0).sum()}/{len(df)}")
# print(f"Avg Obscene reduction:         {(df['obscene_before'] - df['obscene_after']).mean():.4f}")
# print(f"Avg Insult reduction:          {(df['insult_before'] - df['insult_after']).mean():.4f}")
# print(f"Avg Identity attack reduction: {(df['identity_attack_before'] - df['identity_attack_after']).mean():.4f}")
# print("="*55)
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# from google import genai
#
# GEMINI_API_KEY = "AIzaSyBgcL1gL1R9-Hhm7tomr_CC56wTXa6TE_k"
# client = genai.Client(api_key=GEMINI_API_KEY)
#
# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents="Say hello in one word"
# )
# print(response.text)
# # import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# import google.generativeai as genai
# import pandas as pd
# import json
# import re
#
# GEMINI_API_KEY = "AIzaSyBgcL1gL1R9-Hhm7tomr_CC56wTXa6TE_k"
# # genai.configure(api_key=GEMINI_API_KEY)
# # model = genai.GenerativeModel("gemini-1.5-flash")
# from google import genai
# from google.genai import types
#
# client = genai.Client(api_key=GEMINI_API_KEY)
# # Load one toxic row
# df = pd.read_csv("detoxified_evaluated.csv")
# df = df.loc[:, ~df.columns.duplicated()]
# toxic_df = df[df["toxicity_before"] >= 0.5].copy()

# Test on first row only
# row = toxic_df.iloc[0]
# original = str(row["Post_body"])
# detoxified = str(row["detoxified_body"])
#
# print(f"Original: {original}")
# print(f"Detoxified: {detoxified}")
# print()
#
# prompt = f"""You are evaluating a text detoxification system.
#
# Original comment: "{original}"
# Detoxified comment: "{detoxified}"
#
# Rate the detoxification on these dimensions:
#
# 1. TOXICITY_REMOVAL (1-5): Did rewrite remove toxic content?
# 2. MEANING_PRESERVATION (1-5): Is original meaning preserved?
# 3. FLUENCY (1-5): Is output natural and grammatically correct?
# 4. REFUSAL (0 or 1): Did model refuse instead of rewriting?
# 5. OVERALL (1-5): Overall quality
#
# Respond in JSON only:
# {{
#   "toxicity_removal": <1-5>,
#   "meaning_preservation": <1-5>,
#   "fluency": <1-5>,
#   "refusal": <0 or 1>,
#   "overall": <1-5>,
#   "reasoning": "<one sentence>"
# }}"""
#
# # response = model.generate_content(prompt)
# # raw_text = response.text.strip()
# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=prompt
# )
# raw_text = response.text.strip()
#
# print("RAW GEMINI RESPONSE:")
# print(raw_text)
# print()
#
# # Try parsing
# try:
#     cleaned = re.sub(r"```json|```", "", raw_text).strip()
#     result = json.loads(cleaned)
#     print("PARSED SUCCESSFULLY:")
#     print(result)
# except Exception as e:
#     print(f"PARSE ERROR: {e}")
#     print(f"Cleaned text was: {cleaned}")
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# import os
# import time
# import pandas as pd
# import google.generativeai as genai
# import json
# import re
#
# # ── Config ────────────────────────────────────────────────────────────────────
# GEMINI_API_KEY = "AIzaSyBgcL1gL1R9-Hhm7tomr_CC56wTXa6TE_k"  # paste your key here
# TOXICITY_THRESHOLD = 0.5
# DELAY_SECONDS = 4  # keeps us under 15 RPM safely
# INPUT_CSV = "detoxified_evaluated.csv"
# OUTPUT_CSV = "llm_judge_results.csv"
# PROGRESS_CSV = "judge_progress.csv"
#
# # ── Setup Gemini ──────────────────────────────────────────────────────────────
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel("gemini-1.5-flash")
#
# # ── Judge Prompt ──────────────────────────────────────────────────────────────
# def build_prompt(original: str, detoxified: str) -> str:
#     return f"""You are evaluating a text detoxification system.
# Your job is to judge how well a toxic comment was rewritten.
#
# Original comment: "{original}"
# Detoxified comment: "{detoxified}"
#
# Rate the detoxification on these dimensions:
#
# 1. TOXICITY_REMOVAL: Did the rewrite successfully remove toxic content?
#    1 = Still very toxic
#    3 = Partially detoxified
#    5 = Completely non-toxic
#
# 2. MEANING_PRESERVATION: Is the original meaning preserved?
#    1 = Completely different meaning or refusal
#    3 = Some meaning preserved
#    5 = Perfect meaning preservation
#
# 3. FLUENCY: Is the output natural and grammatically correct?
#    1 = Unreadable or refusal
#    3 = Understandable but awkward
#    5 = Perfectly natural
#
# 4. REFUSAL: Did the model refuse instead of rewriting?
#    0 = Yes refused (e.g. "I can't help with that")
#    1 = No actually rewrote the text
#
# 5. OVERALL: Overall quality of detoxification
#    1 = Complete failure
#    3 = Acceptable
#    5 = Perfect
#
# Respond in JSON only with no extra text:
# {{
#   "toxicity_removal": <1-5>,
#   "meaning_preservation": <1-5>,
#   "fluency": <1-5>,
#   "refusal": <0 or 1>,
#   "overall": <1-5>,
#   "reasoning": "<one sentence explanation>"
# }}"""
#
# # ── Judge Function ─────────────────────────────────────────────────────────────
# def judge_row(original: str, detoxified: str) -> dict:
#     try:
#         prompt = build_prompt(original, detoxified)
#         response = model.generate_content(prompt)
#         text = response.text.strip()
#
#         # Clean up response — remove markdown if present
#         text = re.sub(r"```json|```", "", text).strip()
#
#         result = json.loads(text)
#         return result
#
#     except json.JSONDecodeError:
#         # If JSON parsing fails return None scores
#         return {
#             "toxicity_removal": None,
#             "meaning_preservation": None,
#             "fluency": None,
#             "refusal": None,
#             "overall": None,
#             "reasoning": "JSON parse error"
#         }
#     except Exception as e:
#         return {
#             "toxicity_removal": None,
#             "meaning_preservation": None,
#             "fluency": None,
#             "refusal": None,
#             "overall": None,
#             "reasoning": f"Error: {str(e)}"
#         }
#
# # ── Load Data ─────────────────────────────────────────────────────────────────
# print(f"Loading data from {INPUT_CSV}...")
# df = pd.read_csv(INPUT_CSV)
# df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns
#
# # Filter toxic rows only
# toxic_df = df[df["toxicity_before"] >= TOXICITY_THRESHOLD].copy()
# print(f"Total rows: {len(df)}")
# print(f"Toxic rows to judge: {len(toxic_df)}")
# print(f"Estimated time: {len(toxic_df) * DELAY_SECONDS / 60:.1f} minutes")
#
# # ── Check for existing progress ───────────────────────────────────────────────
# start_idx = 0
# results = []
#
# if os.path.exists(PROGRESS_CSV):
#     progress_df = pd.read_csv(PROGRESS_CSV)
#     results = progress_df.to_dict("records")
#     start_idx = len(results)
#     print(f"\nResuming from row {start_idx}...")
# else:
#     print("\nStarting fresh...")
#
# # ── Run Judge ─────────────────────────────────────────────────────────────────
# print("\nRunning LLM Judge...")
# toxic_rows = list(toxic_df.iterrows())
#
# for count, (i, row) in enumerate(toxic_rows[start_idx:], start=start_idx):
#     original   = str(row["Post_body"])
#     detoxified = str(row["detoxified_body"])
#
#     judge_result = judge_row(original, detoxified)
#
#     # Add original row info
#     judge_result["row_index"]        = i
#     judge_result["subreddit"]        = row.get("subreddit", "")
#     judge_result["Post_body"]        = original
#     judge_result["detoxified_body"]  = detoxified
#     judge_result["toxicity_before"]  = row.get("toxicity_before", None)
#     judge_result["toxicity_after"]   = row.get("toxicity_after", None)
#     judge_result["cosine_similarity"]= row.get("cosine_similarity", None)
#     judge_result["bert_score_f1"]    = row.get("bert_score_f1", None)
#     judge_result["is_refusal"]       = row.get("is_refusal", None)
#
#     results.append(judge_result)
#
#     # Save progress every 50 rows
#     if len(results) % 50 == 0:
#         pd.DataFrame(results).to_csv(PROGRESS_CSV, index=False)
#         print(f"  Judged {len(results)}/{len(toxic_df)} rows... (progress saved)")
#
#     # Rate limit delay
#     time.sleep(DELAY_SECONDS)
#
# # ── Save Final Results ────────────────────────────────────────────────────────
# results_df = pd.DataFrame(results)
# results_df.to_csv(OUTPUT_CSV, index=False)
# print(f"\nSaved to {OUTPUT_CSV}")
#
# # ── Summary ───────────────────────────────────────────────────────────────────
# # Drop rows where judge failed
# valid = results_df.dropna(subset=["overall"])
#
# print("\n" + "="*55)
# print("LLM JUDGE SUMMARY")
# print("="*55)
# print(f"Total toxic rows judged:       {len(results_df)}")
# print(f"Valid judgments:               {len(valid)}")
# print(f"Failed judgments:              {len(results_df) - len(valid)}")
# print(f"Avg Toxicity Removal:          {valid['toxicity_removal'].mean():.2f}/5")
# print(f"Avg Meaning Preservation:      {valid['meaning_preservation'].mean():.2f}/5")
# print(f"Avg Fluency:                   {valid['fluency'].mean():.2f}/5")
# print(f"Avg Overall:                   {valid['overall'].mean():.2f}/5")
# print(f"Refusal rate (judge):          {(valid['refusal'] == 0).sum()}/{len(valid)} ({(valid['refusal'] == 0).mean()*100:.1f}%)")
# print("="*55)
#
# # ── Compare Judge vs Automated Metrics ───────────────────────────────────────
# print("\n" + "="*55)
# print("JUDGE vs AUTOMATED METRICS COMPARISON")
# print("="*55)
# print(f"Refusal rate (string match):   34.6%")
# print(f"Refusal rate (LLM judge):      {(valid['refusal'] == 0).mean()*100:.1f}%")
# print(f"Avg Cosine Similarity:         {valid['cosine_similarity'].mean():.4f}")
# print(f"Avg BERTScore F1:              {valid['bert_score_f1'].mean():.4f}")
# print(f"Avg Meaning Preservation:      {valid['meaning_preservation'].mean():.2f}/5")
# print("="*55)
#
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# import os
# import torch
# import pandas as pd
# import json
# import re
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # ── Config ────────────────────────────────────────────────────────────────────
# HF_TOKEN           = os.environ.get("HF_TOKEN")
# JUDGE_MODEL        = "Qwen/Qwen2.5-1.5B-Instruct"
# TOXICITY_THRESHOLD = 0.5
# INPUT_CSV          = "detoxified_evaluated.csv"
# OUTPUT_CSV         = "llm_judge_results.csv"
# PROGRESS_CSV       = "judge_progress.csv"
#
# # ── Load Judge Model ──────────────────────────────────────────────────────────
# print(f"Loading judge model {JUDGE_MODEL}...")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
#
# tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, token=HF_TOKEN)
# model = AutoModelForCausalLM.from_pretrained(
#     JUDGE_MODEL,
#     device_map="auto",
#     dtype=torch.float16,
#     token=HF_TOKEN
# )
# model.eval()
# print("Judge model loaded!")
#
#
#
# # ── Load Data ─────────────────────────────────────────────────────────────────
# print(f"\nLoading data from {INPUT_CSV}...")
#
# # ── Judge Prompt ──────────────────────────────────────────────────────────────
# def build_prompt(original: str, detoxified: str) -> str:
#     return f"""You are evaluating a text detoxification system.
# Your job is to judge how well a toxic comment was rewritten.
#
# Original comment: "{original}"
# Detoxified comment: "{detoxified}"
#
# Rate the detoxification on these dimensions:
#
# 1. TOXICITY_REMOVAL: Did the rewrite successfully remove toxic content?
#    1 = Still very toxic, 3 = Partially detoxified, 5 = Completely non-toxic
#
# 2. MEANING_PRESERVATION: Is the original meaning preserved?
#    1 = Completely different meaning or refusal, 3 = Some meaning, 5 = Perfect
#
# 3. FLUENCY: Is the output natural and grammatically correct?
#    1 = Unreadable, 3 = Understandable but awkward, 5 = Perfectly natural
#
# 4. REFUSAL: Did the model refuse instead of rewriting?
#    0 = Yes refused, 1 = No actually rewrote
#
# 5. OVERALL: Overall quality
#    1 = Complete failure, 3 = Acceptable, 5 = Perfect
#
# Respond in JSON only, no markdown, no extra text:
# {{
#   "toxicity_removal": <1-5>,
#   "meaning_preservation": <1-5>,
#   "fluency": <1-5>,
#   "refusal": <0 or 1>,
#   "overall": <1-5>,
#   "reasoning": "<one sentence>"
# }}"""
#
# # ── Judge Function ────────────────────────────────────────────────────────────
# def judge_row(original: str, detoxified: str) -> dict:
#     try:
#         messages = [
#             {"role": "system", "content": "You are a strict JSON-only evaluation assistant. Always respond with valid JSON only."},
#             {"role": "user", "content": build_prompt(original, detoxified)}
#         ]
#
#         prompt = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 max_new_tokens=200,
#                 do_sample=False,
#                 pad_token_id=tokenizer.eos_token_id,
#                 eos_token_id=tokenizer.eos_token_id,
#             )
#
#         input_length = inputs["input_ids"].shape[-1]
#         new_tokens = outputs[0][input_length:]
#         raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
#
#         # ── Fix: extract JSON from markdown block ──
#         # Remove ```json and ``` markers
#         cleaned = raw_text
#         cleaned = re.sub(r"```json", "", cleaned)
#         cleaned = re.sub(r"```", "", cleaned)
#         cleaned = cleaned.strip()
#
#         # Find the JSON object
#         json_match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
#         if json_match:
#             result = json.loads(json_match.group())
#             return result
#         else:
#             # Try parsing directly
#             result = json.loads(cleaned)
#             return result
#
#     except Exception as e:
#         return {
#             "toxicity_removal":     None,
#             "meaning_preservation": None,
#             "fluency":              None,
#             "refusal":              None,
#             "overall":              None,
#             "reasoning":            f"Error: {str(e)}"
#         }
#
# # ── Load Data ─────────────────────────────────────────────────────────────────
# print(f"\nLoading data from {INPUT_CSV}...")
# df = pd.read_csv(INPUT_CSV)
# df = df.loc[:, ~df.columns.duplicated()]
#
# toxic_df = df[df["toxicity_before"] >= TOXICITY_THRESHOLD].copy()
# print(f"Total rows: {len(df)}")
# print(f"Toxic rows to judge: {len(toxic_df)}")
#
#
#
#
# # ── Check for existing progress ───────────────────────────────────────────────
# start_idx = 0
# results = []
#
# if os.path.exists(PROGRESS_CSV):
#     progress_df = pd.read_csv(PROGRESS_CSV)
#     results = progress_df.to_dict("records")
#     start_idx = len(results)
#     print(f"Resuming from row {start_idx}...")
# else:
#     print("Starting fresh...")
#
# # ── Run Judge ─────────────────────────────────────────────────────────────────
# print("\nRunning LLM Judge...")
# toxic_rows = list(toxic_df.iterrows())
#
# for count, (i, row) in enumerate(toxic_rows[start_idx:], start=start_idx):
#     original   = str(row["Post_body"])
#     detoxified = str(row["detoxified_body"])
#
#     judge_result = judge_row(original, detoxified)
#
#     judge_result["row_index"]         = i
#     judge_result["subreddit"]         = row.get("subreddit", "")
#     judge_result["Post_body"]         = original
#     judge_result["detoxified_body"]   = detoxified
#     judge_result["toxicity_before"]   = row.get("toxicity_before", None)
#     judge_result["toxicity_after"]    = row.get("toxicity_after", None)
#     judge_result["cosine_similarity"] = row.get("cosine_similarity", None)
#     judge_result["bert_score_f1"]     = row.get("bert_score_f1", None)
#     judge_result["is_refusal"]        = row.get("is_refusal", None)
#
#     results.append(judge_result)
#
#     # Save progress every 50 rows
#     if len(results) % 50 == 0:
#         pd.DataFrame(results).to_csv(PROGRESS_CSV, index=False)
#         print(f"  Judged {len(results)}/{len(toxic_df)} rows... (progress saved)")
#
# # ── Save Final Results ────────────────────────────────────────────────────────
# results_df = pd.DataFrame(results)
# results_df.to_csv(OUTPUT_CSV, index=False)
# print(f"\nSaved to {OUTPUT_CSV}")
#
# # ── Summary ───────────────────────────────────────────────────────────────────
# valid = results_df.dropna(subset=["overall"])
#
# print("\n" + "="*55)
# print("LLM JUDGE SUMMARY")
# print("="*55)
# print(f"Total toxic rows judged:       {len(results_df)}")
# print(f"Valid judgments:               {len(valid)}")
# print(f"Failed judgments:              {len(results_df) - len(valid)}")
# print(f"Avg Toxicity Removal:          {valid['toxicity_removal'].mean():.2f}/5")
# print(f"Avg Meaning Preservation:      {valid['meaning_preservation'].mean():.2f}/5")
# print(f"Avg Fluency:                   {valid['fluency'].mean():.2f}/5")
# print(f"Avg Overall:                   {valid['overall'].mean():.2f}/5")
# print(f"Refusal rate (judge):          {(valid['refusal']==0).sum()}/{len(valid)} ({(valid['refusal']==0).mean()*100:.1f}%)")
# print("="*55)
#
# print("\n" + "="*55)
# print("JUDGE vs AUTOMATED METRICS COMPARISON")
# print("="*55)
# print(f"Refusal rate (string match):   34.6%")
# print(f"Refusal rate (LLM judge):      {(valid['refusal']==0).mean()*100:.1f}%")
# print(f"Avg Cosine Similarity:         {valid['cosine_similarity'].mean():.4f}")
# print(f"Avg BERTScore F1:              {valid['bert_score_f1'].mean():.4f}")
# print(f"Avg Meaning Preservation:      {valid['meaning_preservation'].mean():.2f}/5")
# print("="*55)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import re
import json
import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Auth ──────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN")

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
JUDGE_MODEL        = "Qwen/Qwen2.5-3B-Instruct"
CSV_PATH           = "detoxified_comments.csv"
OUTPUT_CSV         = "detoxified_evaluated.csv"
JUDGE_OUTPUT_CSV   = "llm_judge_results.csv"
JUDGE_PROGRESS_CSV = "judge_progress.csv"
TOXICITY_THRESHOLD = 0.5

REFUSAL_PHRASES = [
    "i can't help",
    "i can't fulfill",
    "i cannot assist",
    "i can't assist",
    "i cannot provide",
    "i can't provide",
    "i'm not able",
    "i cannot help",
    "i can't do",
    "i won't",
]

# ══════════════════════════════════════════════════════
# PART 1 — AUTOMATED EVALUATOR
# ══════════════════════════════════════════════════════
class DetoxEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print("Loading similarity model...")
        self.sim_model = SentenceTransformer(
            EMBEDDING_MODEL, device=str(self.device)
        )

        print("Loading toxicity model...")
        self.tox_model = Detoxify('original', device=self.device)

        print("Loading ROUGE scorer...")
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True
        )

    def cosine_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.sim_model.encode(text1, convert_to_tensor=True)
        emb2 = self.sim_model.encode(text2, convert_to_tensor=True)
        return F.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()

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

    def is_refusal(self, text: str) -> bool:
        text_lower = text.lower().strip()
        return any(phrase in text_lower for phrase in REFUSAL_PHRASES)

    def length_ratio(self, original: str, detoxified: str) -> float:
        orig_len = len(original.split())
        if orig_len == 0:
            return 0.0
        return len(detoxified.split()) / orig_len

    def bleu(self, original: str, detoxified: str) -> float:
        reference = [original.lower().split()]
        hypothesis = detoxified.lower().split()
        smoothie = SmoothingFunction().method1
        return sentence_bleu(
            reference, hypothesis,
            smoothing_function=smoothie
        )

    def rouge_score(self, original: str, detoxified: str) -> float:
        scores = self.rouge_scorer.score(original, detoxified)
        return scores["rougeL"].fmeasure

    def evaluate(self, original: str, detoxified: str) -> dict:
        sim        = self.cosine_similarity(original, detoxified)
        tox_before = self.toxicity_scores(original)
        tox_after  = self.toxicity_scores(detoxified)
        refusal    = self.is_refusal(detoxified)
        length_r   = self.length_ratio(original, detoxified)
        bleu_s     = self.bleu(original, detoxified)
        rouge_s    = self.rouge_score(original, detoxified)

        return {
            "cosine_similarity":          sim,
            "bleu_score":                 bleu_s,
            "rouge_l":                    rouge_s,
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
            "is_refusal":                 refusal,
            "length_ratio":               length_r,
        }


# ══════════════════════════════════════════════════════
# PART 2 — LLM JUDGE
# ══════════════════════════════════════════════════════
class LLMJudge:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading judge model {JUDGE_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            JUDGE_MODEL, token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL,
            device_map="auto",
            dtype=torch.float16,
            token=HF_TOKEN
        )
        self.model.eval()
        print("Judge model loaded!")

    def build_prompt(self, original: str, detoxified: str) -> str:
        return f"""You are evaluating a text detoxification system.
Your job is to judge how well a toxic comment was rewritten.

Original comment: "{original}"
Detoxified comment: "{detoxified}"

Rate the detoxification on these dimensions:

1. TOXICITY_REMOVAL: Did the rewrite successfully remove toxic content?
   1 = Still very toxic, 3 = Partially detoxified, 5 = Completely non-toxic

2. MEANING_PRESERVATION: Is the original meaning preserved?
   1 = Completely different meaning or refusal, 3 = Some meaning, 5 = Perfect

3. FLUENCY: Is the output natural and grammatically correct?
   1 = Unreadable, 3 = Understandable but awkward, 5 = Perfectly natural

4. REFUSAL: Did the model refuse instead of rewriting?
   0 = Yes refused, 1 = No actually rewrote

5. OVERALL: Overall quality
   1 = Complete failure, 3 = Acceptable, 5 = Perfect

Respond in JSON only, no markdown, no extra text:
{{
  "toxicity_removal": <1-5>,
  "meaning_preservation": <1-5>,
  "fluency": <1-5>,
  "refusal": <0 or 1>,
  "overall": <1-5>,
  "reasoning": "<one sentence>"
}}"""

    def judge(self, original: str, detoxified: str) -> dict:
        try:
            messages = [
                {"role": "system", "content": "You are a strict JSON-only evaluation assistant. Always respond with valid JSON only."},
                {"role": "user", "content": self.build_prompt(original, detoxified)}
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[-1]
            new_tokens = outputs[0][input_length:]
            raw_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            # Clean markdown
            cleaned = re.sub(r"```json|```", "", raw_text).strip()
            json_match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(cleaned)

        except Exception as e:
            return {
                "toxicity_removal":     None,
                "meaning_preservation": None,
                "fluency":              None,
                "refusal":              None,
                "overall":              None,
                "reasoning":            f"Error: {str(e)}"
            }


# ══════════════════════════════════════════════════════
# MAIN — Load Data
# ══════════════════════════════════════════════════════
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(subset=["Post_body", "detoxified_body"])
df = df[df["Post_body"].str.strip() != ""]
df = df[df["detoxified_body"].str.strip() != ""]
print(f"Rows after cleaning: {len(df)}")

# ══════════════════════════════════════════════════════
# PART 1 — Run Automated Evaluation
# ══════════════════════════════════════════════════════
evaluator = DetoxEvaluator()

print("\nRunning automated evaluation...")
results = []
for i, row in df.iterrows():
    result = evaluator.evaluate(
        str(row["Post_body"]),
        str(row["detoxified_body"])
    )
    results.append(result)
    if len(results) % 100 == 0:
        print(f"  Evaluated {len(results)}/{len(df)} rows...")

# BERTScore
print("\nComputing BERTScore...")
originals   = df["Post_body"].tolist()
detoxifieds = df["detoxified_body"].tolist()
P, R, F1 = bert_score_fn(
    detoxifieds, originals, lang="en", verbose=True
)

eval_df = pd.DataFrame(results, index=df.index)
eval_df["bert_score_f1"] = F1.numpy()
df = pd.concat([df, eval_df], axis=1)
df = df.loc[:, ~df.columns.duplicated()]
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved automated evaluation to {OUTPUT_CSV}")

# ── Automated Summary ─────────────────────────────────
print("\n" + "="*55)
print("AUTOMATED EVALUATION SUMMARY")
print("="*55)
print(f"Total rows evaluated:          {len(df)}")
print(f"Refusal rate:                  {df['is_refusal'].sum()}/{len(df)} ({df['is_refusal'].mean()*100:.1f}%)")
print(f"Avg Length Ratio:              {df['length_ratio'].mean():.4f}")
print(f"Avg Cosine Similarity:         {df['cosine_similarity'].mean():.4f}")
print(f"Avg BERTScore F1:              {df['bert_score_f1'].mean():.4f}")
print(f"Avg BLEU Score:                {df['bleu_score'].mean():.4f}")
print(f"Avg ROUGE-L:                   {df['rouge_l'].mean():.4f}")
print(f"Avg Toxicity Before:           {df['toxicity_before'].mean():.4f}")
print(f"Avg Toxicity After:            {df['toxicity_after'].mean():.4f}")
print(f"Avg Toxicity Change:           {df['toxicity_change'].mean():.4f}")
print(f"Rows toxicity reduced:         {(df['toxicity_change'] < 0).sum()}/{len(df)}")
print(f"Rows toxicity increased:       {(df['toxicity_change'] > 0).sum()}/{len(df)}")
print(f"Avg Obscene reduction:         {(df['obscene_before'] - df['obscene_after']).mean():.4f}")
print(f"Avg Insult reduction:          {(df['insult_before'] - df['insult_after']).mean():.4f}")
print(f"Avg Identity attack reduction: {(df['identity_attack_before'] - df['identity_attack_after']).mean():.4f}")
print("="*55)

# ══════════════════════════════════════════════════════
# PART 2 — Run LLM Judge on Toxic Rows Only
# ══════════════════════════════════════════════════════
toxic_df = df[df["toxicity_before"] >= TOXICITY_THRESHOLD].copy()
print(f"\nToxic rows to judge: {len(toxic_df)}")

judge = LLMJudge()

# Check for existing progress
start_idx = 0
judge_results = []

if os.path.exists(JUDGE_PROGRESS_CSV):
    progress_df = pd.read_csv(JUDGE_PROGRESS_CSV)
    judge_results = progress_df.to_dict("records")
    start_idx = len(judge_results)
    print(f"Resuming from row {start_idx}...")
else:
    print("Starting fresh...")

print("\nRunning LLM Judge...")
toxic_rows = list(toxic_df.iterrows())

for count, (i, row) in enumerate(toxic_rows[start_idx:], start=start_idx):
    original   = str(row["Post_body"])
    detoxified = str(row["detoxified_body"])

    judge_result = judge.judge(original, detoxified)

    # Add all metrics
    judge_result["row_index"]              = i
    judge_result["subreddit"]              = row.get("subreddit", "")
    judge_result["Post_body"]              = original
    judge_result["detoxified_body"]        = detoxified
    judge_result["toxicity_before"]        = row.get("toxicity_before", None)
    judge_result["toxicity_after"]         = row.get("toxicity_after", None)
    judge_result["toxicity_change"]        = row.get("toxicity_change", None)
    judge_result["severe_toxicity_before"] = row.get("severe_toxicity_before", None)
    judge_result["severe_toxicity_after"]  = row.get("severe_toxicity_after", None)
    judge_result["obscene_before"]         = row.get("obscene_before", None)
    judge_result["obscene_after"]          = row.get("obscene_after", None)
    judge_result["insult_before"]          = row.get("insult_before", None)
    judge_result["insult_after"]           = row.get("insult_after", None)
    judge_result["identity_attack_before"] = row.get("identity_attack_before", None)
    judge_result["identity_attack_after"]  = row.get("identity_attack_after", None)
    judge_result["cosine_similarity"]      = row.get("cosine_similarity", None)
    judge_result["bert_score_f1"]          = row.get("bert_score_f1", None)
    judge_result["bleu_score"]             = row.get("bleu_score", None)
    judge_result["rouge_l"]               = row.get("rouge_l", None)
    judge_result["length_ratio"]           = row.get("length_ratio", None)
    judge_result["is_refusal"]             = row.get("is_refusal", None)

    judge_results.append(judge_result)

    if len(judge_results) % 50 == 0:
        pd.DataFrame(judge_results).to_csv(JUDGE_PROGRESS_CSV, index=False)
        print(f"  Judged {len(judge_results)}/{len(toxic_df)} rows... (progress saved)")

# Save final judge results
judge_df = pd.DataFrame(judge_results)
judge_df.to_csv(JUDGE_OUTPUT_CSV, index=False)
print(f"\nSaved LLM judge results to {JUDGE_OUTPUT_CSV}")

# ── LLM Judge Summary ─────────────────────────────────
valid = judge_df.dropna(subset=["overall"])

print("\n" + "="*55)
print("LLM JUDGE SUMMARY — TOXIC ROWS ONLY")
print("="*55)
print(f"Total toxic rows judged:       {len(judge_df)}")
print(f"Valid judgments:               {len(valid)}")
print(f"Failed judgments:              {len(judge_df) - len(valid)}")
print()
print("── LLM Judge Scores ──────────────────────────────")
print(f"Avg Toxicity Removal:          {valid['toxicity_removal'].mean():.2f}/5")
print(f"Avg Meaning Preservation:      {valid['meaning_preservation'].mean():.2f}/5")
print(f"Avg Fluency:                   {valid['fluency'].mean():.2f}/5")
print(f"Avg Overall:                   {valid['overall'].mean():.2f}/5")
print(f"Refusal rate (judge):          {(valid['refusal']==0).sum()}/{len(valid)} ({(valid['refusal']==0).mean()*100:.1f}%)")
print()
print("── Automated Metrics (toxic rows only) ───────────")
print(f"Avg Cosine Similarity:         {valid['cosine_similarity'].mean():.4f}")
print(f"Avg BERTScore F1:              {valid['bert_score_f1'].mean():.4f}")
print(f"Avg BLEU Score:                {valid['bleu_score'].mean():.4f}")
print(f"Avg ROUGE-L:                   {valid['rouge_l'].mean():.4f}")
print(f"Avg Length Ratio:              {valid['length_ratio'].mean():.4f}")
print(f"Refusal rate (string match):   {valid['is_refusal'].mean()*100:.1f}%")
print()
print("── Toxicity Metrics (toxic rows only) ────────────")
print(f"Avg Toxicity Before:           {valid['toxicity_before'].mean():.4f}")
print(f"Avg Toxicity After:            {valid['toxicity_after'].mean():.4f}")
print(f"Avg Toxicity Change:           {valid['toxicity_change'].mean():.4f}")
print(f"Avg Obscene reduction:         {(valid['obscene_before'] - valid['obscene_after']).mean():.4f}")
print(f"Avg Insult reduction:          {(valid['insult_before'] - valid['insult_after']).mean():.4f}")
print(f"Avg Identity attack reduction: {(valid['identity_attack_before'] - valid['identity_attack_after']).mean():.4f}")
print()
print("── Judge vs Automated Comparison ─────────────────")
print(f"Refusal rate (string match):   {valid['is_refusal'].mean()*100:.1f}%")
print(f"Refusal rate (LLM judge):      {(valid['refusal']==0).mean()*100:.1f}%")
print(f"Cosine Similarity:             {valid['cosine_similarity'].mean():.4f}")
print(f"BERTScore F1:                  {valid['bert_score_f1'].mean():.4f}")
print(f"Meaning Preservation (judge):  {valid['meaning_preservation'].mean():.2f}/5")
print("="*55)