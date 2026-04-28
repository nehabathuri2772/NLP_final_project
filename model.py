import torch
from torch import nn
from transformers import AutoTokenizer
from trl.experimental.ppo import AutoModelForCausalLMWithValueHead

from constants import GENERATION_MODEL, GENERATION_CONFIG

class DetoxificationModel:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model & tokenizer
        print(f"Loading detox model {GENERATION_MODEL} on {self.device}...")
        self._load_model()
        self.model.eval()

    def _load_model(self):
        # TODO: Add check for local LoRA model
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(GENERATION_MODEL).to(self.device)

        # Add new items to gen config
        GENERATION_CONFIG.device = self.device
        GENERATION_CONFIG.pad_token_id = self.tokenizer.eos_token_id
        GENERATION_CONFIG.eos_token_id = self.tokenizer.eos_token_id


    def get_system_prompt(self):
        return "You are a helpful assistant. Rewrite the following text so that it" \
                " is completely non-toxic, polite, and preserves its original meaning" \
                " as much as possible. Keep the word length similar to the original response." \
                " Only output the rewritten text, nothing else."

    def get_user_prompt(self, text):
        return f"Rewrite this text: '{text}'"

    def build_prompt(self, toxic_text):
        conversation = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_user_prompt(toxic_text)}
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt_str

    def detoxify(self, text: str):
        # Build instruction prompt
        prompt = self.build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GENERATION_CONFIG,
            )

        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        input_length = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0][input_length:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "prompt": prompt,
            "completion": completion,
            "raw_response": raw_response,
        }

class AggregatorModel(nn.Module):
    # Small MLP for Aggregation

    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)