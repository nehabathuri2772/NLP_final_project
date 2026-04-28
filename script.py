import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from detoxify import Detoxify
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ── Auth ──────────────────────────────────────────────────────────────────────
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN
# paste your new token here

# ── Config ────────────────────────────────────────────────────────────────────
GENERATION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
NUM_ROWS = 10000

TOXICITY_THRESHOLD = 0.5

# ── Detoxification Model ──────────────────────────────────────────────────────
class DetoxificationModel:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading detox model {GENERATION_MODEL} ...")
        self._load_model()
        self.model.eval()

    def _load_model(self):
        token = os.environ["HF_TOKEN"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            GENERATION_MODEL,
            token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL,
            device_map="auto",
            dtype=torch.float16,
            token=token
        )

    def get_system_prompt(self):
        return (
            "You are a text detoxifier. "
            "Rewrite the input text to be non-toxic and polite. "
            "Output ONLY the final rewritten text. "
            "Do NOT include any introduction, explanation, or preamble. "
            "Do NOT start with 'Here is', 'Here's', 'Rewritten', or similar phrases. "
            "Just output the clean rewritten sentence and nothing else."
            "If the text is already polite, return it as is. "
            "If the text is very short or just one word, return it as is. "
            "Never refuse. Always return a rewritten or original version. "
        )

    def detoxify(self, text: str):
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": text}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0][input_length:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip any leftover preamble lines
        lines = completion.split("\n")
        skip_keywords = ["here's", "here is", "rewritten", "non-toxic", "following", "below"]
        clean_lines = [
            line for line in lines
            if not any(kw in line.lower() for kw in skip_keywords)
        ]
        completion = " ".join(clean_lines).strip()

        return completion

# ── Load Dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset(
    "fddemarco/pushshift-reddit-comments",
    split="train",
    streaming=True
)

rows = []
for item in ds.take(NUM_ROWS):
    rows.append({
        "subreddit":        item.get("subreddit", ""),
        "subreddit_id":     item.get("subreddit_id", ""),
        "post_id":          item.get("link_id", ""),
        "post_author":      item.get("author", ""),
        "Post_body":        item.get("body", ""),
        "post_created_utc": item.get("created_utc", ""),
        "detoxified_body":  None
    })

df = pd.DataFrame(rows)
print(f"DataFrame loaded! Shape: {df.shape}")
print(df.head(3))

# ── Filter Toxic Comments ─────────────────────────────────────────────────────
print("Filtering toxic comments...")
filter_model = Detoxify('original')

toxic_indices = []
for i, row in df.iterrows():
    text = str(row["Post_body"])
    if text.strip():
        score = filter_model.predict(text)["toxicity"]
        df.at[i, "toxicity_before"] = score
        if score >= TOXICITY_THRESHOLD:
            toxic_indices.append(i)
    if (i + 1) % 100 == 0:
        print(f"  Filtered {i + 1}/{NUM_ROWS} rows... Found {len(toxic_indices)} toxic so far")

toxic_df = df.loc[toxic_indices].copy()
print(f"\nFound {len(toxic_indices)} toxic comments out of {NUM_ROWS} total ({len(toxic_indices)/NUM_ROWS*100:.1f}%)")

# ── Run Detoxification ────────────────────────────────────────────────────────
model = DetoxificationModel()

print("\nRunning detoxification...")
for i, row in df.iterrows():
    original_text = row["Post_body"]
    if original_text and original_text.strip():
        df.at[i, "detoxified_body"] = model.detoxify(original_text)
    else:
        df.at[i, "detoxified_body"] = ""

    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{NUM_ROWS} rows...")

# ── Save & Preview ────────────────────────────────────────────────────────────
df.to_csv("detoxified_comments.csv", index=False)
print("\nDone! Saved to detoxified_comments.csv")
print(df[["Post_body", "detoxified_body"]].head(5).to_string())