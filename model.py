import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

GENERATION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
GENERATION_CONFIG = GenerationConfig(
    max_new_tokens=2048,
    do_sample=False, # No sampling since this is research work, needs consistent results
    temperature=0.
)

class DetoxificationModel:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model & tokenizer
        print(f"Loading detox model {GENERATION_MODEL} ...")
        self._load_model()
        self.model.eval()

    def _load_model(self):
        # TODO: Add check for local LoRA model
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL, device_map="auto")


    def get_system_prompt(self):
        return f"""
        You are a helpful assistant. Rewrite the following text so that it
        is completely non-toxic, polite, and preserves its original meaning
        as much as possible. Only output the rewritten text, nothing else.
        """

    def get_user_prompt(self, text):
        return f"Text: {text}"

    def detoxify(self, text: str):
        # Build instruction prompt
        prompt = {"system": self.get_system_prompt(), "user": self.get_user_prompt(text)}
        inputs = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GENERATION_CONFIG,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = raw_response[len(prompt):].strip()

        return {
            "prompt": prompt,
            "completion": completion,
            "raw_response": raw_response,
        }

class AggregatorModel:
    # TODO: Design aggregator to take in stats from detoxification and compute score
    def __init__(self):
        pass