# LIGN167: Finetuning Llama-3.1-8B-Instruct on MedQA Dataset

**Authors**: Fong Vo, Brian Huynh, Hao Zhang, Brian Liu, Sarah He  
**Date**: November 2025

## Project Summary
 This repository evaluates multiple adaptation techniques for improving Llama-3.1-8B-Instruct performance on the MedQA medical question-answering benchmark. We compare baseline inference, IA³ parameter-efficient fine-tuning, system prompting, and chain-of-thought (CoT) reasoning strategies, with a focus on both accuracy and calibration metrics.

This study investigates whether adaptation techniques—including IA³ fine-tuning, system prompting, and chain-of-thought reasoning—can improve Llama-3.1-8B-Instruct performance on the MedQA medical question-answering benchmark. We find that the baseline model achieves 74.07% accuracy. While IA³ fine-tuning yields no improvement, inference-time strategies reveal a critical trade-off: system prompting with chain-of-thought reasoning reduces accuracy slightly (72.14%) but dramatically improves calibration (ECE: 0.3379 → 0.1083). For clinical applications requiring reliable confidence estimates, CoT-based reasoning offers a safer deployment strategy than parameter-efficient fine-tuning.

## Contents

- **`cot/`** – Chain-of-Thought evaluation code and results
  - `system_prompt_cot.py` – Evaluates CoT reasoning + system prompting approach
  - `cot_generations.jsonl` – Detailed predictions and reasoning traces
  - `cot.log` – Evaluation logs

- **`system_prompt.py`** – System prompting baseline evaluation

- **`playground.ipynb`** – Experimental notebook for prototyping

---

**Repository Structure**:
```
lign167/
├── README.md                      # This file
├── system_prompt.py               # System prompting baseline
├── cot/
│   ├── system_prompt_cot.py       # CoT evaluation (with full documentation)
│   ├── cot_generations.jsonl      # Results with reasoning traces
│   └── cot.log                    # Evaluation logs
└── playground.ipynb               # Experimental notebook
```
