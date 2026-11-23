"""
Local LLM inference client (spec §2, §4). Loads the configured model once
(4-bit NF4 quantization) and exposes a single generate() call that takes a
system+user prompt pair and returns the raw text plus exact token counts.

extract_code() lives in code_extraction.py, not here, so modules that only
need to extract code (orchestrator.py, tests) don't have to import torch.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.code_extraction import extract_code  # re-exported for convenience

__all__ = ["LLMClient", "extract_code"]


class LLMClient:
    def __init__(self, model_name: str, device_map: str = "cuda:0",
                 temperature: float = 0.2, max_new_tokens: int = 1024, seed=None):
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        # Sampling (temperature > 0) is random; seeding once at startup makes the
        # whole run's sequence of draws reproducible. None = unseeded (non-reproducible).
        if seed is not None:
            torch.manual_seed(seed)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config, device_map=device_map,
        )
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Returns {"text": str, "prompt_tokens": int, "generated_tokens": int}."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(generated_ids.shape[0]),
        }


