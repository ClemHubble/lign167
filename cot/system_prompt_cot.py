"""
Chain-of-Thought (CoT) Evaluation Framework for Medical QA with LLaMA

This script evaluates the performance of a medical LLM (Meta Llama-3.1-8B-Instruct)
on the MedQA dataset. It uses chain-of-thought prompting where the model
generates reasoning before predicting an answer. Confidence scores are computed using
negative log-likelihood, and calibration metrics (accuracy and ECE) are reported.

Key Components:
- Model: meta-llama/Llama-3.1-8B-Instruct
- Dataset: MedQA (medical multiple-choice questions)
- Evaluation Metric: Accuracy and Expected Calibration Error (ECE)
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
import json
from tqdm import tqdm

# Print GPU availability for debugging
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

def expected_calibration_error(confidences, correctness, num_bins=15):
    """
    Calculate Expected Calibration Error (ECE) to measure calibration of model predictions.
    
    ECE measures the difference between model confidence and actual correctness across
    different confidence bins. A well-calibrated model has ECE close to 0.
    
    Args:
        confidences (list or np.ndarray): Model confidence scores for each prediction (0-1).
        correctness (list or np.ndarray): Binary correctness labels (1=correct, 0=incorrect).
        num_bins (int): Number of bins to partition confidence scores. Default=15.
        
    Returns:
        float: Expected Calibration Error value (0-1, lower is better).
        
    Algorithm:
        1. Partition predictions into num_bins based on confidence scores
        2. For each bin, compute average confidence and average correctness
        3. Weight each bin's error by its proportion in the data
        4. Return weighted sum of absolute differences
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    ece = 0.0
    bins = np.linspace(0, 1, num_bins + 1)

    for i in range(num_bins):
        lower, upper = bins[i], bins[i+1]
        # Get indices of predictions in this confidence bin
        idx = (confidences >= lower) & (confidences < upper)
        if np.sum(idx) == 0:
            continue
        # Calculate average confidence and accuracy in this bin
        bin_conf = np.mean(confidences[idx])
        bin_acc  = np.mean(correctness[idx])
        # Add weighted calibration error for this bin
        ece     += np.abs(bin_conf - bin_acc) * np.mean(idx)

    return ece

from huggingface_hub import login
# TODO: Add Hugging Face API token for gated model access
#login("")

# Model configuration
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with half precision (float16) to reduce memory usage
# device_map="auto" automatically distributes model across available GPUs
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set model to evaluation mode (disables dropout, batch norm updates, etc.)
model.eval()
print("Model loaded.")

# Load MedQA dataset (English version with 4 options)
print("Loading MedQA...")
ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa")["train"]

# Preprocess dataset to match answer text with multiple-choice options
processed = []

for q, opts, ans_text in zip(ds["question"], ds["choices"], ds["answer"]):
    
    # ans_text is a list containing the answer text (e.g., ["Nitrofurantoin"])
    correct_text = ans_text[0]

    # Find the index of the correct answer in the choices list
    try:
        correct_idx = opts.index(correct_text)
    except ValueError:
        # Skip data points where the answer doesn't match any choice (data quality issue)
        continue

    # Convert numerical index to letter (0→A, 1→B, 2→C, 3→D)
    correct_letter = chr(ord("A") + correct_idx)
    
    processed.append({
        "question": q,
        "options": opts,
        "answer": correct_letter
    })

def format_query(question, choices):
    """
    Format a medical multiple-choice question with system instructions for chain-of-thought.
    
    This creates a prompt that instructs the model to:
    1. Act as a board-certified physician
    2. Reason through the problem step-by-step
    3. Provide explicit reasoning before selecting an answer
    
    Args:
        question (str): The medical question text.
        choices (list): List of answer options (typically 4 for MedQA).
        
    Returns:
        str: Formatted prompt with system instruction, question, choices, and reasoning prefix.
    """
    # System prompt that establishes physician persona and chain-of-thought strategy
    system_prompt = "You are a board-certified physician with deep expertise in internal medicine. You answer clinical questions using evidence-based guidelines, pathophysiology, and differential diagnosis. Carefully analyze the question, eliminate incorrect options, and choose the BEST answer based strictly on medical knowledge. If the question lacks sufficient information, state your assumption explicitly. Never fabricate facts or conditions not stated in the question. First, think step-by-step and provide your reasoning. Then, respond with the letter corresponding to the correct answer."

    text = system_prompt
    text += f"Question: {question}\n"
    
    # Format answer choices as A, B, C, D, etc.
    for i, ch in enumerate(choices):
        letter = chr(ord('A') + i)
        text += f"{letter}. {ch}\n"
    
    # Add prefix to prompt model to start reasoning
    text += "Reasoning:"
    return text


def score_option(model, tokenizer, prompt, letter):
    """
    Score an answer option using negative log-likelihood.
    
    This function computes how "likely" a given answer is by calculating the model's
    loss when predicting the answer given the prompt. The loss is only computed on
    the answer tokens, not the prompt tokens (masked with -100).
    
    Higher scores indicate the model thinks this answer is more likely/better.
    This is used to rank and select among the four answer options.
    
    Args:
        model: The language model (with generate and forward pass capabilities).
        tokenizer: The tokenizer for encoding text.
        prompt (str): The prompt context (question + choices + reasoning prefix).
        letter (str): The answer letter to score (e.g., "A", "B", "C", "D").
        
    Returns:
        float: Negative log-likelihood score (negated so higher = better).
        
    Note:
        Uses negative NLL as the score so that argmax can select the best option.
    """
    # Format the complete answer statement
    answer_text = f"Answer: {letter}"

    # Tokenize prompt and answer separately
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    answer_ids = tokenizer(answer_text, return_tensors="pt").input_ids.to(model.device)

    # Concatenate: [prompt_tokens] + [answer_tokens]
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    # Create label mask: only compute loss on answer tokens, not prompt tokens
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100   # -100 tokens are ignored in loss computation

    # Compute loss (only on answer tokens)
    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
        nll = out.loss.item()

    return -nll   # Negate so higher score = better prediction



# Initialize evaluation tracking variables
correct_count = 0
confidences = []   # Model confidence for each prediction
correctness = []   # Correctness (0/1) for each prediction
logs = []          # Detailed logs for each question

print("Running evaluation...")

# Main evaluation loop
for item in tqdm(processed, desc="Evaluating", ncols=80):
    
    question = item["question"]
    options = item["options"]
    correct_answer_letter = item["answer"]
    correct_idx = ord(correct_answer_letter) - ord("A")  # Convert A/B/C/D to 0/1/2/3

    # Step 1: Build prompt with physician persona and chain-of-thought instruction
    prompt = format_query(question, options)
    
    # Step 2: Generate model's reasoning (chain-of-thought)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,          # Allow up to 512 tokens for reasoning
            do_sample=False,             # Greedy decoding for reproducibility
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract reasoning context (up to "Answer:" if present)
    full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    if "Answer:" in full_output:
        # Extract everything up to and including the first "Answer:" marker
        parts = full_output.split("Answer:")
        reasoning_context = parts[0] + "Answer:"
    else:
        # If model didn't generate "Answer:" prompt, append it
        reasoning_context = full_output + "\nAnswer:"
    
    # Step 3: Score each answer option (A, B, C, D) using negative log-likelihood
    letters = ["A", "B", "C", "D"]
    scores = [
        score_option(model, tokenizer, reasoning_context, letter)
        for letter in letters
    ]

    # Step 4: Select the highest-scoring option
    pred_idx = int(np.argmax(scores))
    
    # Step 5: Compute confidence using softmax over the scores
    conf = float(torch.softmax(torch.tensor(scores), dim=0)[pred_idx])
    
    # Step 6: Check if prediction is correct
    is_correct = (pred_idx == correct_idx)

    # Update tracking variables
    correct_count += is_correct
    confidences.append(conf)
    correctness.append(is_correct)
    
    # Log detailed results for this question
    logs.append({
        "question": question,
        "prompt": prompt,
        "generated_reasoning": reasoning_context,
        "correct_answer": correct_answer_letter,
        "predicted_answer": letters[pred_idx],
        "is_correct": bool(is_correct)
    })


# Compute and report evaluation metrics
accuracy = correct_count / len(processed)
ece = expected_calibration_error(confidences, correctness)

print("----- RESULTS -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"ECE:      {ece:.4f}")

# Save detailed logs to JSONL file (one JSON object per line)
with open("cot_generations.jsonl", "w") as f:
    for entry in logs:
        f.write(json.dumps(entry) + "\n")
