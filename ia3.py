import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import IA3Config, get_peft_model
import numpy as np
from tqdm import tqdm

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
# LOAD MODEL + APPLY IA3 PEFT
# -------------------------
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print("Loading base model with FP16 on GPU...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.gradient_checkpointing_enable()  # saves memory
model.eval()

# Apply IA3
peft_config = IA3Config()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# -------------------------
# LOAD DATASET
# -------------------------
print("Loading MedQA dataset...")
ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_bigbio_qa", trust_remote_code=True)

train_ds = ds["train"]
eval_ds = ds["validation"] if "validation" in ds else ds["test"]

# -------------------------
# FORMAT DATA
# -------------------------
def format_training_example(example):
    text = f"Question: {example['question']}\n"
    for i, ch in enumerate(example["choices"]):
        letter = chr(ord("A") + i)
        text += f"{letter}. {ch}\n"
    text += f"Answer: {example['answer'][0]}"  # first element of answer list
    return {"text": text}

train_ds = train_ds.map(format_training_example)
eval_ds  = eval_ds.map(format_training_example)

tokenized_train = train_ds.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
    batched=True
)

tokenized_eval = eval_ds.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512),
    batched=True
)

# -------------------------
# TRAINING ARGUMENTS
# -------------------------
out_dir = "./llama8b_medqa_ia3"

training_args = TrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=1,  # reduce memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # effective batch size = 4
    fp16=True,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none"  # disable wandb if memory is tight
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# -------------------------
# TRAIN
# -------------------------
print("Starting fine-tuning...")
trainer.train()
model.save_pretrained(out_dir)

# -------------------------
# EVALUATION LOOP (memory-efficient)
# -------------------------
def format_eval_prompt(question, choices):
    txt = f"Question: {question}\n"
    for i, ch in enumerate(choices):
        letter = chr(ord("A") + i)
        txt += f"{letter}. {ch}\n"
    txt += "Answer: "
    return txt

correct = 0
confidences = []
correctness = []

option_tokens = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in ["A", "B", "C", "D"]]

for item in tqdm(eval_ds, desc="Evaluating"):
    prompt = format_eval_prompt(item["question"], item["choices"])
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True
        )

    logits = out.scores[0]
    probs = torch.softmax(logits, dim=-1)
    option_probs = [probs[0, tok].item() for tok in option_tokens]

    pred_idx = int(np.argmax(option_probs))
    confidence = option_probs[pred_idx]
    correct_idx = next(i for i, ch in enumerate(item["choices"]) if ch.strip().lower() == item["answer"][0].strip().lower())

    is_correct = pred_idx == correct_idx
    correct += is_correct
    correctness.append(is_correct)
    confidences.append(confidence)

accuracy = correct / len(eval_ds)
ece = expected_calibration_error(confidences, correctness)

print("----- RESULTS (FINETUNED MODEL) -----")
print(f"Accuracy: {accuracy:.4f}")
print(f"ECE:      {ece:.4f}")
