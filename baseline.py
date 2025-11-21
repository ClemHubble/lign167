import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# -------------------------
# ECE CALCULATION
# -------------------------
def expected_calibration_error(confidences, correctness, num_bins=15):
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    ece = 0.0
    bins = np.linspace(0, 1, num_bins + 1)

    for i in range(num_bins):
        lower, upper = bins[i], bins[i+1]
        idx = (confidences >= lower) & (confidences < upper)
        if idx.sum() == 0:
            continue
        bin_conf = np.mean(confidences[idx])
        bin_acc = np.mean(correctness[idx])
        ece += np.abs(bin_conf - bin_acc) * np.mean(idx)
    return ece

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading Llama 3.1 8B Instruct on GPU...")
model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # automatically uses GPU if available
)
model.eval()

# -------------------------
# LOAD DATA
# -------------------------
print("Loading MedQA dataset...")
ds = load_dataset(
    "bigbio/med_qa",
    "med_qa_en_4options_bigbio_qa",
    trust_remote_code=True
)["train"]

print("Dataset size:", len(ds))

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def format_prompt(question, choices):
    txt = f"Question: {question}\n"
    for i, ch in enumerate(choices):
        letter = chr(ord("A") + i)
        txt += f"{letter}. {ch}\n"
    txt += "Answer: "
    return txt

def get_label(answer_text_list, choices):
    """
    answer_text_list: list of one string (the correct answer)
    choices: list of 4 option strings
    """
    answer_text = answer_text_list[0]  # take the first element
    normalized_answer = answer_text.strip().lower()
    for idx, choice in enumerate(choices):
        if choice.strip().lower() == normalized_answer:
            return idx
    raise ValueError(f"Answer '{answer_text}' not found in choices {choices}")


# -------------------------
# EVALUATION LOOP
# -------------------------
batch_size = 32
confidences = []
correctness = []

# Precompute option tokens for "A", "B", "C", "D"
option_tokens = [
    tokenizer.encode(letter, add_special_tokens=False)[0]
    for letter in ["A", "B", "C", "D"]
]

print("Running evaluation...")

for start in range(0, len(ds), batch_size):
    end = min(start + batch_size, len(ds))
    batch = ds.select(range(start, end))

    prompts = [
        format_prompt(batch[i]["question"], batch[i]["choices"])
        for i in range(len(batch))
    ]

    correct_indices = [
        get_label(batch[i]["answer"], batch[i]["choices"])
        for i in range(len(batch))
    ]

    enc = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True
        )

    # logits for the generated token
    logits = out.scores[0]
    probs = torch.softmax(logits, dim=-1)

    # get probability for each option token
    option_probs = torch.stack([probs[:, tok] for tok in option_tokens], dim=1)

    preds = torch.argmax(option_probs, dim=1).tolist()
    batch_conf = torch.max(option_probs, dim=1).values.tolist()

    for i in range(len(batch)):
        is_correct = preds[i] == correct_indices[i]
        correctness.append(is_correct)
        confidences.append(batch_conf[i])

# -------------------------
# RESULTS
# -------------------------
accuracy = sum(correctness) / len(correctness)
ece = expected_calibration_error(confidences, correctness)

print("----- RESULTS -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"ECE:      {ece:.4f}")
