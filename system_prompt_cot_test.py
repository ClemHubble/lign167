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

sample_processed = processed[:5]

for item in tqdm(sample_processed, desc="Evaluating", ncols=80):

    question = item["question"]
    options = item["options"]
    correct_answer_letter = item["answer"]
    correct_idx = ord(correct_answer_letter) - ord("A")

    # Build prompt WITH doctor persona
    prompt = format_query(question, options)
    
    # Generate reasoning
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # We want to generate until "Answer:" or max tokens
    # But since we can't easily stop exactly at "Answer:" without custom stopping criteria or post-processing,
    # we'll generate a bit and then append "Answer:" manually if not present, or just trust the model flow.
    # A better approach for CoT + Scoring is:
    # 1. Generate reasoning.
    # 2. Append "Answer:" to the generated text.
    # 3. Score A/B/C/D.
    
    with torch.no_grad():
        # Generate reasoning tokens
        # We use a stop sequence or just max new tokens. 
        # Let's generate up to 256 tokens for reasoning.
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False, # Greedy decoding for reproducibility/stability
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # We need to construct the prompt for scoring. 
    # The model might have already outputted "Answer: X". We need to be careful.
    # If we want to force-score the options, we should take the generated reasoning 
    # and append "Answer:" if it's not the last thing.
    
    # However, simply taking the full output might include the answer if the model was eager.
    # Let's try to strip any "Answer: ..." from the end if present, or just use the full output as context
    # PROVIDED it ends with "Answer:" or we append it.
    
    # Let's assume the model followed instructions and outputted reasoning.
    # We'll append "\nAnswer:" to the generated text (or ensure it's there) and then score.
    
    # A robust way:
    # 1. Take `full_output`.
    # 2. Check if "Answer:" is in it.
    #    If yes, truncate everything after "Answer:".
    #    If no, append "\nAnswer:".
    
    if "Answer:" in full_output:
        # Truncate to keep context up to Answer:
        # We want the prompt passed to score_option to end with "Answer: " 
        # so that score_option appends "A" and we score "Answer: A".
        
        # Split by "Answer:" and take the first part + "Answer:"
        parts = full_output.split("Answer:")
        reasoning_context = parts[0] + "Answer:"
    else:
        reasoning_context = full_output + "\nAnswer:"
        
    # Now score options based on this reasoning context
    letters = ["A", "B", "C", "D"]

    # Score each answer choice using full log-likelihood
    scores = [
        score_option(model, tokenizer, reasoning_context, letter)
        for letter in letters
    ]

    # Choose highest scoring answer
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

# Save logs
with open("cot_generations_test.jsonl", "w") as f:
    for entry in logs:
        f.write(json.dumps(entry) + "\n")


accuracy = correct_count / len(sample_processed)
ece = expected_calibration_error(confidences, correctness)

print("----- RESULTS -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"ECE:      {ece:.4f}")
