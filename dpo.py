import argparse
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

def main(args):
    """
    Main function to execute the DPO training pipeline.
    This script performs DPO fine-tuning. It can do so using a full model update
    or a parameter-efficient fine-tuning approach with LoRA.
    """
    # --- 1. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")

    # Quantization configuration for memory efficiency
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Load the base model to be fine-tuned (policy model)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load the reference model (frozen)
    # If no path is provided, a copy of the policy model is used before LoRA is applied
    if args.ref_model_name_or_path:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        ref_model = None # DPOTrainer will create a reference model from the policy model

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Configure LoRA if enabled, otherwise prepare for full fine-tuning ---
    if args.use_lora:
        print("Configuring LoRA for parameter-efficient fine-tuning...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(',') if args.lora_target_modules else None,
        )
        # Apply LoRA to the policy model. The reference model remains a full model.
        policy_model = get_peft_model(policy_model, peft_config)
        print("LoRA model configured. Trainable parameters:")
        policy_model.print_trainable_parameters()
    else:
        print("Proceeding with full model fine-tuning. LoRA is not enabled.")
        peft_config = None


    # --- 3. Load and Prepare the Dataset ---
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    # Split dataset if a validation split is specified
    if args.validation_split_percentage > 0:
        train_test_split = dataset["train"].train_test_split(test_size=args.validation_split_percentage / 100.0)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = None

    print(f"Training on {len(train_dataset)} samples.")
    if eval_dataset:
        print(f"Evaluating on {len(eval_dataset)} samples.")


    # --- 4. Set up Training Arguments ---
    # Hyperparameters that were previously command-line arguments are now hardcoded here.
    # You can modify these values directly in the script.
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,          # Hardcoded value
        per_device_eval_batch_size=4,           # Hardcoded value
        gradient_accumulation_steps=4,          # Hardcoded value
        learning_rate=5e-7,                     # Hardcoded value
        lr_scheduler_type="cosine",             # Hardcoded value
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        bf16=True, # Recommended for Ampere GPUs
        tf32=True, # Recommended for Ampere GPUs
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=False,
    )

    # --- 5. Initialize the DPOTrainer ---
    # DPO-specific hyperparameters are also hardcoded here now.
    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config, # Pass peft_config; trainer handles LoRA or full tuning
        max_prompt_length=512,  
        max_length=1024,        
        loss_type="sigmoid",    
    )

    # --- 6. Start Training ---
    print("Starting DPO training...")
    dpo_trainer.train()

    # --- 7. Save the Final Model ---
    print(f"Saving final model to {args.output_dir}")
    dpo_trainer.save_model(args.output_dir)

    # Save the tokenizer
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DPO training script. Supports full fine-tuning or LoRA.")

    # Model and Tokenizer Arguments
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the base model for DPO.")
    parser.add_argument("--ref_model_name_or_path", type=str, default=None, help="Path to the reference model. If None, a copy of the base model is used.")

    # Dataset Arguments
    parser.add_argument("--dataset_name", type=str, default="trl-lib/ultrafeedback_binarized", help="Name of the preference dataset on Hugging Face Hub.")
    parser.add_argument("--validation_split_percentage", type=int, default=5, help="Percentage of the dataset to use for validation.")

    # Training Hyperparameters
    parser.add_argument("--output_dir", type=str, default="./dpo_finetuned_model", help="Directory to save the final model.")
    parser.add_argument("--beta", type=float, default=0.1, help="The beta hyperparameter for the DPO loss.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If set, overrides num_train_epochs.")

    # Logging and Saving Arguments
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save a checkpoint every N steps.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Run evaluation every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total number of saved checkpoints.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging to Weights & Biases.")

    # LoRA Arguments
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA for parameter-efficient fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated list of module names to apply LoRA to (e.g., 'q_proj,v_proj').")

    # Quantization Arguments
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization with BitsAndBytes.")

    args = parser.parse_args()
    main(args)

# Example command to run with LoRA:
# python advanced_dpo_script.py \
#     --model_name_or_path "meta-llama/Llama-2-7b-hf" \
#     --dataset_name "trl-lib/ultrafeedback_binarized" \
#     --output_dir "./llama2-7b-dpo-lora" \
#     --use_lora \
#     --use_4bit \
#     --lora_r 64 \
#     --lora_alpha 16 \
#     --beta 0.1 \
#     --num_train_epochs 1 \
#     --logging_steps 10 \
#     --save_steps 200 \
#     --use_wandb

# Example command to run with full fine-tuning (no --use_lora flag):
# python advanced_dpo_script.py \
#     --model_name_or_path "meta-llama/Llama-2-7b-hf" \
#     --dataset_name "trl-lib/ultrafeedback_binarized" \
#     --output_dir "./llama2-7b-dpo-full" \
#     --use_4bit \
#     --beta 0.1 \
#     --num_train_epochs 1 \
#     --logging_steps 10 \
#     --save_steps 200 \
#     --use_wandb
