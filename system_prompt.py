import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
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
#login("") #TODO add Hugging Face Token

model_name = "meta-llama/Llama-3.1-8B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

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
    system_prompt = "You are a board-certified physician with deep expertise in internal medicine. You answer clinical questions using evidence-based guidelines, pathophysiology, and differential diagnosis. Carefully analyze the question, eliminate incorrect options, and choose the BEST answer based strictly on medical knowledge. If the question lacks sufficient information, state your assumption explicitly. Never fabricate facts or conditions not stated in the question. Respond ONLY with the letter corresponding to the correct answer."

    text = system_prompt
    text += f"Question: {question}\n"
    
    for i, ch in enumerate(choices):
        letter = chr(ord('A') + i)
        text += f"{letter}. {ch}\n"
    
    text += "Answer:"
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

    # Compute loss 
    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
        nll = out.loss.item()

    return -nll 

correct_count = 0
confidences = []
correctness = []

print("Running evaluation...")

for item in tqdm(processed, desc="Evaluating", ncols=80):

    question = item["question"]
    options = item["options"]
    correct_answer_letter = item["answer"]
    correct_idx = ord(correct_answer_letter) - ord("A")

    # Build prompt WITH doctor persona
    prompt = format_query(question, options)
    
    letters = ["A", "B", "C", "D"]

    # Score each answer choice using full log-likelihood
    scores = [
        score_option(model, tokenizer, prompt, letter)
        for letter in letters
    ]

    # Choose highest scoring answer
    pred_idx = int(np.argmax(scores))
    conf = float(torch.softmax(torch.tensor(scores), dim=0)[pred_idx])
    is_correct = (pred_idx == correct_idx)

    correct_count += is_correct
    confidences.append(conf)
    correctness.append(is_correct)


accuracy = correct_count / len(processed)
ece = expected_calibration_error(confidences, correctness)

print("----- RESULTS -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"ECE:      {ece:.4f}")
