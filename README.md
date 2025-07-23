# Advanced Direct Preference Optimization (DPO) Training Script

This repository contains a sophisticated Python script for fine-tuning Large Language Models (LLMs) using **Direct Preference Optimization (DPO)**. The script is designed to be flexible and efficient, offering support for both full-model fine-tuning and parameter-efficient fine-tuning with **LoRA**.

---

## Overview

**Direct Preference Optimization (DPO)** is a stable and computationally efficient method for aligning LLMs with human or AI-generated preferences. It offers a direct alternative to more complex reinforcement learning-based approaches like RLHF (*Reinforcement Learning from Human Feedback*).

This script leverages the **Hugging Face ecosystem**, particularly the `transformers`, `peft`, and `trl` libraries, to provide a robust implementation of the DPO algorithm. It allows users to train models on preference datasets, where each sample consists of:

- A prompt
- A *chosen* (preferred) response
- A *rejected* (dispreferred) response

---

## Key Features

- **Direct Preference Optimization**  
  Implements the core DPO loss function for direct alignment on preference data.

- **Flexible Fine-Tuning**  
  Supports both:
  - **Full Fine-Tuning**: Updates all parameters of the language model.
  - **Parameter-Efficient Fine-Tuning (PEFT)**: Uses **LoRA** (Low-Rank Adaptation) to drastically reduce the number of trainable parameters and memory requirements.

- **Quantization**  
  Integrates **4-bit quantization** via `bitsandbytes` to load and train large models on consumer-grade hardware.

- **Ease of Use**  
  Controlled via a clear command-line interface with sensible defaults.

- **Customizable**  
  Key hyperparameters for training and DPO are well-documented and can be easily modified.

- **Monitoring**  
  Includes optional support for logging metrics to **Weights & Biases (wandb)** for experiment tracking.

---

## Prerequisites

Before running the script, ensure you have Python installed along with the necessary libraries. You can install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## How the Script Works

The script follows a logical pipeline from model loading to saving the final fine-tuned artifact:

1. **Argument Parsing**  
   Parses command-line arguments to configure the training run.

2. **Model & Tokenizer Loading**  
   Loads the specified base model (policy) and its tokenizer. Optionally loads a reference model. If none is provided, a frozen copy of the base model is used.  
   4-bit quantization can be enabled with the `--use_4bit` flag.

3. **LoRA Configuration (Optional)**  
   If `--use_lora` is present, applies a LoRA adapter to freeze the base model and only train adapter layers.

4. **Dataset Preparation**  
   Loads the specified preference dataset from Hugging Face and splits it into training/validation sets.

5. **Trainer Initialization**  
   Sets up `TrainingArguments` and initializes `DPOTrainer`, which handles loss computation, training loop, and evaluation.

6. **Training**  
   Starts training using:

   ```python
   dpo_trainer.train()
   ```

7. **Model Saving**  
   Saves the final model (full or LoRA adapter) and tokenizer to the specified output directory.

---

## Usage Examples

### Example 1: DPO Fine-Tuning with LoRA and 4-bit Quantization

This command fine-tunes the `Llama-2-7b-hf` model using LoRA with a rank of 64 and 4-bit quantization.

```bash
python advanced_dpo_script.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --dataset_name "trl-lib/ultrafeedback_binarized" \
    --output_dir "./llama2-7b-dpo-lora" \
    --use_lora \
    --use_4bit \
    --lora_r 64 \
    --lora_alpha 16 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 200 \
    --use_wandb
```

---

### Example 2: Full DPO Fine-Tuning

This command performs full fine-tuning of the `Llama-2-7b-hf` model (no LoRA). It updates all model parameters and requires significantly more GPU memory.

```bash
python advanced_dpo_script.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --dataset_name "trl-lib/ultrafeedback_binarized" \
    --output_dir "./llama2-7b-dpo-full" \
    --use_4bit \
    --beta 0.1 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 200 \
    --use_wandb
```

---
