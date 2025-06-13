from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import os

# === Paths ===
base_model_path = "/path/to/base_model"  # e.g., EuroLLM-1.7B
lora_checkpoint_path = "/path/to/checkpoint-1000"  # e.g., checkpoint dir with adapter_model.bin
output_path = "/path/to/merged_model"  # where to save the merged full model

# === Load base model and PEFT model ===
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
print("Loading PEFT (LoRA) adapter...")
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
#Make sure the model and adapter are loaded with the same torch_dtype.

# === Merge LoRA weights into base model ===
print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

# === Save merged model ===
print(f"Saving merged model to {output_path} ...")
model.save_pretrained(output_path)

# === Optionally save tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print("Done.")
