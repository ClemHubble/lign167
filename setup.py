from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading MedQA dataset...")

load_dataset("araag2/MedQA", 'processed', split="train")

print("Downloading Llama 3.1 8B model...")
model = "meta-llama/Llama-3.1-8B-Instruct"

AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)