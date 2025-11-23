import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
from tqdm import tqdm

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

def expected_calibration_error(confidences, correctness, num_bins=15):
    """
    Computes the Expected Calibration Error (ECE) for a set of predictions.

    The ECE measures how well a model's predicted confidence scores match the
    true empirical accuracies. Predictions are grouped into bins over the
    interval [0, 1], and the discrepancy between average confidence and
    average accuracy in each bin is weighted by the bin's relative size.

    Args:
        confidences: a list or array of predicted confidence scores in [0, 1].
        correctness: a list or array of binary indicators where 1 denotes a
                     correct prediction and 0 denotes an incorrect one.
        num_bins:    the number of equally sized bins used to partition the
                     confidence interval, default is 15.

    Returns:
        the expected calibration error computed over the given predictions.
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
    """
    Formats a multiple-choice clinical question into a standardized prompt
    conditioned with a system-level instruction for medical reasoning.

    The function prepends a detailed system prompt that frames the model as a
    board-certified physician and enforces evidence-based, non-hallucinatory
    reasoning. It then appends the question text, enumerates the provided
    answer choices using letter labels, and ends with an
    "Answer:" tag to signal where the model should output its final choice.

    Args:
        question: a string containing the clinical question to be answered.
        choices:  a list of answer-choice strings to be labeled and included
                  in the formatted prompt.

    Returns:
        a single formatted string containing the system prompt, question,
        labeled choices, and an answer field for model inference.
    """
    system_prompt = "You are a board-certified physician with deep expertise in internal medicine. You answer clinical questions using evidence-based guidelines, pathophysiology, and differential diagnosis. Carefully analyze the question, eliminate incorrect options, and choose the BEST answer based strictly on medical knowledge. If the question lacks sufficient information, state your assumption explicitly. Never fabricate facts or conditions not stated in the question. Respond ONLY with the letter corresponding to the correct answer."

    text = system_prompt
    text += f"Question: {question}\n"
    
    for i, ch in enumerate(choices):
        letter = chr(ord('A') + i)
        text += f"{letter}. {ch}\n"
    
    text += "Answer:"
    return text


def score_option(model, tokenizer, prompt, letter):
    """
    Computes a scalar score for a candidate answer option by evaluating the
    model's negative log-likelihood (NLL) of generating that option given a
    formatted prompt.

    The function appends an answer string of the form ``"Answer: <letter>"`` to
    the prompt, tokenizes both components, and constructs a full input sequence.
    All prompt tokens are masked in the label tensor (set to –100) so that the
    loss is computed **only** over the answer tokens. The returned score is the
    negative of the NLL, so higher values indicate that the model assigns higher
    probability to the candidate answer.

    Args:
        model:     a Hugging Face causal language model used to compute the
                   conditional log-likelihood of the answer.
        tokenizer: the tokenizer associated with the model, used to encode both
                   the prompt and the answer text.
        prompt:    a string representing the full question prompt (including
                   system instruction, question, and choices).
        letter:    a character specifying the answer option to score.

    Returns:
        a float representing the negative log-likelihood of the model generating
        the given answer option conditioned on the prompt.
    """

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
