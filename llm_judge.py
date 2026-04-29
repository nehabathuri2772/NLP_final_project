import json
import re
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from constants import JUDGE_MODEL

class LLMJudge:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\tLoading LLM Judge Model ({JUDGE_MODEL})...")
        self.tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)

        # Use bnb config to make it take less space
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, device_map="auto", quantization_config=bnb_config)
        self.model.eval()
        print("\tJudge model loaded!")

    # Build prompt
    def build_prompt(self, original: str, detoxified: str) -> str:
        return f"""You are evaluating a text detoxification system. Compare these two comments to see if toxicity was removed while preserving meaning.

Original comment: "{original}"
Detoxified comment: "{detoxified}"

Rate:

1. TOXICITY_REMOVAL (1-5) - If toxicity was truly removed
2. MEANING_PRESERVATION (1-5) - If the meaning of the detoxified comment matches what the original comment was saying
3. FLUENCY (1-5) - If the detoxified comment flows like the original text behavior
4. REFUSAL (0 or 1) - If the detoxified comment has words like I'm sorry or refusal patterns
5. OVERALL (1-5) - Overall score if all of the above were achieved

Respond ONLY in JSON:
{{
  "toxicity_removal": <1-5>,
  "meaning_preservation": <1-5>,
  "fluency": <1-5>,
  "refusal": <0 or 1>,
  "overall": <1-5>,
  "reasoning": "<one sentence>"
}}"""

    # Generate function
    def _generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are evaluating a text detoxification system. Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse JSON output
    def _parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```json|```", "", text).strip()

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group()

        return json.loads(cleaned)

    # Judges an example comparing original and detoxified repsonse
    def judge(self, original: str, detoxified: str) -> Dict[str, Any]:
        try:
            # Build prompt and generate response for LLM judge metrics
            prompt = self.build_prompt(original, detoxified)
            raw = self._generate(prompt)
            return self._parse_json(raw)

        except Exception as e:
            print("\tLLM Judge Error:", str(e))
            return {
                "toxicity_removal": None,
                "meaning_preservation": None,
                "fluency": None,
                "refusal": None,
                "overall": None,
                "reasoning": f"Error: {str(e)}",
            }