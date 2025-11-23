import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
import json
from tqdm import tqdm

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

def expected_calibration_error(confidences, correctness, num_bins=15):
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

from huggingface_hub import login
login("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")

model_name = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
# model.config.pad_token_id = model.config.eos_token_id


model.eval()
print("Model loaded.")

print("Loading MedQA...")

ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa")["train"]

processed = []

for q, opts, ans_text in zip(ds["question"], ds["choices"], ds["answer"]):
    
    # ans_text is a list like ["Nitrofurantoin"]
    correct_text = ans_text[0]

    # find the index of the correct answer in choices
    try:
        correct_idx = opts.index(correct_text)
    except ValueError:
        # skip weird data
        continue

    # convert index → letter
    correct_letter = chr(ord("A") + correct_idx)
    
    processed.append({
        "question": q,
        "options": opts,
        "answer": correct_letter
    })

def format_query(question, choices):
    system_prompt = "You are a board-certified physician with deep expertise in internal medicine. You answer clinical questions using evidence-based guidelines, pathophysiology, and differential diagnosis. Carefully analyze the question, eliminate incorrect options, and choose the BEST answer based strictly on medical knowledge. If the question lacks sufficient information, state your assumption explicitly. Never fabricate facts or conditions not stated in the question. First, think step-by-step and provide your reasoning. Then, respond with the letter corresponding to the correct answer."

    text = system_prompt
    text += f"Question: {question}\n"
    
    for i, ch in enumerate(choices):
        letter = chr(ord('A') + i)
        text += f"{letter}. {ch}\n"
    
    text += "Reasoning:"
    return text


def score_option(model, tokenizer, prompt, letter):
    # Build answer text
    answer_text = f"Answer: {letter}"

    # Tokenize prompt and answer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    answer_ids = tokenizer(answer_text, return_tensors="pt").input_ids.to(model.device)

    # Build full sequence: [prompt] + [answer]
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    # Create labels: ignore prompt tokens with -100
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100   # mask out prompt tokens

    # Compute loss (only on answer tokens)
    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
        nll = out.loss.item()

    return -nll   # higher = better



correct_count = 0
confidences = []
correctness = []
logs = []

print("Running evaluation...")

# sample_processed = processed[:30]
# Using full processed list as per original file intent (though it had sample_processed commented out or not used in the original view? 
# Wait, the original file view showed `sample_processed = processed[:30]` was NOT present in the loop in `system_prompt_cot.py`?
# Let's check the `view_file` output for `system_prompt_cot.py` again.
# Ah, I see `sample_processed = processed[:30]` in `system_prompt_cot.py` in the previous `view_file` output (Step 66).
# Wait, Step 66 output shows `sample_processed = processed[:30]` at line 120. 
# So `system_prompt_cot.py` IS also a test file or at least restricted? 
# The user said "modifying @[lign167/system_prompt_cot.py] and @[lign167/system_prompt_cot_test.py]".
# Usually `_test` implies a smaller subset. 
# If `system_prompt_cot.py` also has `[:30]`, I should probably remove that restriction if it's meant to be the "real" run, 
# OR just keep it as is if I'm not sure. 
# However, the user didn't ask to change the dataset size, just to add logging.
# I will keep the logic as close to the current file as possible, just adding CoT and logging.
# BUT wait, `system_prompt_cot.py` in Step 66 ALREADY HAS CoT logic?
# Let's re-examine Step 66.
# Line 77: "First, think step-by-step..." -> YES, it has CoT prompt.
# Line 132: "# Generate reasoning" -> YES, it has generation logic.
# So `system_prompt_cot.py` WAS ALREADY MODIFIED?
# Ah, I might have misread the file path in previous turns or the user provided a file that already had some changes?
# Or maybe I am hallucinating.
# Let's look at Step 66 again.
# File Path: `file:///workspace/lign167/system_prompt_cot.py`
# It DOES have the CoT changes.
# Did I edit it?
# I only edited `system_prompt_cot_test.py` in Step 21.
# Maybe the user copied the content?
# Or maybe `system_prompt_cot.py` was already like that?
# Wait, Step 66 was a `view_file` call I made in Step 65.
# It shows the file content.
# It seems `system_prompt_cot.py` ALREADY has the CoT implementation.
# That's strange if I didn't edit it.
# Unless... `system_prompt_cot.py` IS `system_prompt_cot_test.py`?
# No, they are different files.
# Let's check `system_prompt_cot_test.py` content from Step 4.
# Step 4 `system_prompt_cot_test.py` did NOT have CoT (Line 77 was standard prompt).
# I edited `system_prompt_cot_test.py` in Step 21.
# So `system_prompt_cot_test.py` has CoT.
# Now Step 66 shows `system_prompt_cot.py`.
# It HAS CoT.
# Why?
# Maybe the user edited it? "The following changes were made by the USER..."
# Step 61 showed user changes to `system_prompt_cot_test.py`.
# I don't see user changes to `system_prompt_cot.py`.
# Maybe I am misreading Step 66.
# Let's look closely at Step 66.
# Line 77: "...First, think step-by-step..."
# Yes.
# Line 120: `sample_processed = processed[:30]`
# Line 122: `for item in tqdm(sample_processed...`
# Okay, so `system_prompt_cot.py` seems to be a copy of `system_prompt_cot_test.py` or similar.
# If it already has CoT, I just need to add logging.
# AND I need to make sure `system_prompt_cot_test.py` has logging.
#
# Wait, if `system_prompt_cot.py` has `[:30]`, it's basically a test file too.
# I will add logging to `system_prompt_cot.py` as well.
#
# I will use `import json` which is missing in `system_prompt_cot.py` (Step 66 shows imports at top, no json).

import json

for item in tqdm(sample_processed, desc="Evaluating", ncols=80):

    question = item["question"]
    options = item["options"]
    correct_answer_letter = item["answer"]
    correct_idx = ord(correct_answer_letter) - ord("A")

    # Build prompt WITH doctor persona
    prompt = format_query(question, options)
    
    # Generate reasoning
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    if "Answer:" in full_output:
        parts = full_output.split("Answer:")
        reasoning_context = parts[0] + "Answer:"
    else:
        reasoning_context = full_output + "\nAnswer:"
        
    letters = ["A", "B", "C", "D"]

    scores = [
        score_option(model, tokenizer, reasoning_context, letter)
        for letter in letters
    ]

    pred_idx = int(np.argmax(scores))
    conf = float(torch.softmax(torch.tensor(scores), dim=0)[pred_idx])
    is_correct = (pred_idx == correct_idx)

    correct_count += is_correct
    confidences.append(conf)
    correctness.append(is_correct)
    
    logs.append({
        "question": question,
        "prompt": prompt,
        "generated_reasoning": reasoning_context,
        "correct_answer": correct_answer_letter,
        "predicted_answer": letters[pred_idx],
        "is_correct": bool(is_correct)
    })


accuracy = correct_count / len(sample_processed)
ece = expected_calibration_error(confidences, correctness)

print("----- RESULTS -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"ECE:      {ece:.4f}")

with open("cot_generations.jsonl", "w") as f:
    for entry in logs:
        f.write(json.dumps(entry) + "\n")
