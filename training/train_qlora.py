import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# Use a standard relative import
from map_facts_to_text import row_to_example

def to_chat(example):
    """Converts a dataset row to the required prompt-response chat format."""
    ex = row_to_example(example)
    return {"text": ex["prompt"] + "\n" + ex["response"]}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Hugging Face model identifier.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the full dataset JSONL file.")
    parser.add_argument("--val_split_size", type=float, default=0.1, help="Proportion of the dataset to use for validation (e.g., 0.1 for 10%%).")
    parser.add_argument("--output_dir", type=str, default="./llama3-3b-aml-qlora", help="Directory to save the trained model adapter.")
    args = parser.parse_args()

    print(f"Starting training with the following configuration:")
    print(f"  Model: {args.model_id}")
    print(f"  Dataset file: {args.dataset_file}")
    print(f"  Validation split size: {args.val_split_size}")
    print(f"  Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the full dataset and split it into training and validation sets
    full_dataset = load_dataset("json", data_files=args.dataset_file, split="train")
    split_dataset = full_dataset.train_test_split(test_size=args.val_split_size, shuffle=True, seed=42)

    ds_train = split_dataset["train"].map(to_chat, remove_columns=["facts","decision","fired_rules"])
    ds_val   = split_dataset["test"].map(to_chat, remove_columns=["facts","decision","fired_rules"])

    model_load_kwargs = {"device_map": "auto"}
    if torch.backends.mps.is_available():
        model_load_kwargs["device_map"] = {"": "mps"}
        model_load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_load_kwargs,
    )

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tok.pad_token = tok.eos_token

    try:
        import tensorboard  # noqa: F401
        report_to = ["tensorboard"]
    except ImportError:
        report_to = []

    lora = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_length=512,
        packing=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        dataset_text_field="text",
        report_to=report_to,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        args=cfg,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
