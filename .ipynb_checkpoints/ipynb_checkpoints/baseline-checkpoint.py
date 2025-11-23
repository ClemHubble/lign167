import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math

# -------------------------
# ECE CALCULATION
# -------------------------
def expected_calibration_error(confidences, correctness, num_bins=15):
    """
    confidences: list/array of predicted probability for chosen option
    correctness: list/array of 0/1 indicating correct prediction
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    ece = 0.0
    bins = np.linspace(0, 1, num_bins + 1)

    for i in range(num_bins):
        lower, upper = bins[i], bins[i+1]
        idx = (confidences >= lower) & (confidences < upper)
        if np.sum(idx) == 0:
            continue
        bin_conf = np.mean(confidences[idx])
        bin_acc  = np.mean(correctness[idx])
        ece     += np.abs(bin_conf - bin_acc) * np.mean(idx)

    return ece


# -------------------------
# LOAD MODEL
# -------------------------
print("Loading Llama 3.1 8B Instruct baseline...")
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -------------------------
# LOAD DATA
# -------------------------
print("Loading MedQA...")
ds = load_dataset("bigbio/med_qa", "med_qa_en")["train"]  

# -------------------------
# EVALUATION LOOP
# -------------------------
correct = 0
confidences = []
correctness = []

def format_query(question, choices):
    text = f"Question: {question}\n"
    for i, ch in enumerate(choices):
        letter = chr(ord('A') + i)
        text += f"{letter}. {ch}\n"
    text += "Answer:"
    return text

print("Running evaluation...")

for item in ds:
    question = item["question"]
    options = item["options"]
    correct_answer_letter = item["answer"]  # e.g. "A", "B", "C", "D"
    correct_idx = ord(correct_answer_letter) - ord("A")

    prompt = format_query(question, options)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Extract logprobs of options
    last_token = outputs.sequences[0, -1]
    logits = outputs.scores[-1][0]
    probs = torch.softmax(logits, dim=-1)

    # get probability for each answer letter token
    option_tokens = [
        tokenizer.encode(letter, add_special_tokens=False)[0]
        for letter in ["A", "B", "C", "D"]
    ]
    option_probs = [probs[token].item() for token in option_tokens]

    pred_idx = int(np.argmax(option_probs))
    confidence = option_probs[pred_idx]

    is_correct = (pred_idx == correct_idx)

    correct += is_correct
    confidences.append(confidence)
    correctness.append(is_correct)

accuracy = correct / len(ds)
ece = expected_calibration_error(confidences, correctness)

print("----- RESULTS (BASELINE) -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"ECE:      {ece:.4f}")
