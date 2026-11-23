import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant_config, device_map="cuda:0")
inputs = tok("Say hello in one word.", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=10)
print("OUTPUT:", tok.decode(out[0], skip_special_tokens=True))