import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import IA3Config, get_peft_model
import numpy as np

# -------------------------------------------------------
# ECE CALCULATION (same as baseline)
# -------------------------------------------------------
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


# -------------------------------------------------------
# LOAD MODEL + APPLY PEFT METHOD
# -------------------------------------------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# IA3 (switchable)
peft_config = IA3Config()

print("Applying IA3 PEFT...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------
print("Loading MedQA...")
ds = load_dataset("bigbio/med_qa", "med_qa_en")

train_ds = ds["train"]
eval_ds  = ds["validation"] if "validation" in ds else ds["test"]

# -------------------------------------------------------
# PREPARE TRAINING FORMAT
# Convert MCQ into instruction → answer string pairs
# -------------------------------------------------------
def format_training_example(example):
    text = f"Question: {example['question']}\n"
    for i, ch in enumerate(example["options"]):
        letter = chr(ord("A") + i)
        text += f"{letter}. {ch}\n"
    text += f"Answer: {example['answer']}"
    return {"text": text}

train_ds = train_ds.map(format_training_example)
eval_ds  = eval_ds.map(format_training_example)

tokenized_train = train_ds.map(
    lambda e: tokenizer(e["text"], truncation=True),
    batched=True
)

tokenized_eval = eval_ds.map(
    lambda e: tokenizer(e["text"], truncation=True),
    batched=True
)

# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------
out_dir = "./llama8b_medqa_ia3"

training_args = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=True,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)

trainer = torch.cuda.amp.autocast(enabled=True)(
    None
)  # prevents torch autocast override issues

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("Starting fine-tuning...")
trainer.train()

model.save_pretrained(out_dir)

# -------------------------------------------------------
# EVALUATION (same logic as baseline)
# -------------------------------------------------------
print("Running evaluation...")

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

for item in eval_ds:
    question = item["question"]
    options = item["options"]
    correct_answer_letter = item["answer"]
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

    last_token = outputs.sequences[0, -1]
    logits = outputs.scores[-1][0]
    probs = torch.softmax(logits, dim=-1)

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

accuracy = correct / len(eval_ds)
ece = expected_calibration_error(confidences, correctness)

print("----- RESULTS (FINETUNED MODEL) -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"ECE:      {ece:.4f}")
