import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from pathlib import Path

GCS_SCHEME = "gs://"
HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)
SUPPORTED_FLASH_ATTN_IMPLEMENTATIONS = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn",
    "kernels-community/flash-attn3",
    "kernels-community/vllm-flash-attn3",
}

# Use a standard relative import
from map_facts_to_text import row_to_example

def to_chat(example):
    """Converts a dataset row to the required prompt-response chat format."""
    ex = row_to_example(example)
    return {"text": ex["prompt"] + "\n" + ex["response"]}

def resolve_hf_token(arg_token: str | None) -> str:
    """Resolve the Hugging Face Hub token from CLI args or environment."""
    if arg_token:
        return arg_token

    for key in HF_TOKEN_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value

    raise RuntimeError(
        "Missing Hugging Face token. Pass --hf_token or set one of: "
        + ", ".join(HF_TOKEN_ENV_KEYS)
    )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Hugging Face model identifier.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the full dataset JSONL file.")
    parser.add_argument("--val_split_size", type=float, default=0.1, help="Proportion of the dataset to use for validation (e.g., 0.1 for 10%%).")
    parser.add_argument("--output_dir", type=str, default="./llama3-3b-aml-qlora", help="Directory to save the trained model adapter.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum number of tokens per packed example.")
    parser.add_argument("--packing", action="store_true", help="Enable dataset packing (requires flash attention kernels).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Per-device eval batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging frequency in steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency in steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint save frequency in steps.")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", help="Disable gradient checkpointing.")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision where supported.")
    parser.add_argument("--upload_output_to", type=str, default=None, help="Optional gs:// URI to upload the trained adapter artifacts.")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional Hugging Face Hub token for gated models.")
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

    hf_token = resolve_hf_token(args.hf_token)
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    model_load_kwargs = {"device_map": "auto"}
    if torch.backends.mps.is_available():
        model_load_kwargs["device_map"] = {"": "mps"}
        model_load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        **model_load_kwargs,
    )

    tok = AutoTokenizer.from_pretrained(args.model_id, token=hf_token, use_fast=True)
    tok.pad_token = tok.eos_token

    packing_enabled = args.packing
    if args.packing:
        attn_impl = getattr(getattr(model, "config", None), "attn_implementation", None)
        if attn_impl not in SUPPORTED_FLASH_ATTN_IMPLEMENTATIONS:
            print(
                "Packing requested but attention implementation "
                f"{attn_impl!r} is not one of {sorted(SUPPORTED_FLASH_ATTN_IMPLEMENTATIONS)}. "
                "Disabling packing to avoid flash-attn dependency issues."
            )
            packing_enabled = False

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
        max_length=args.max_length,
        packing=packing_enabled,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        dataset_text_field="text",
        report_to=report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
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

    if args.upload_output_to:
        upload_directory_to_gcs(args.output_dir, args.upload_output_to)

def upload_directory_to_gcs(local_dir: str, gcs_uri: str) -> None:
    """Upload the contents of `local_dir` to the target GCS URI."""
    if not gcs_uri.startswith(GCS_SCHEME):
        raise ValueError("upload_output_to must be a gs:// URI")

    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError("google-cloud-storage is required to upload to GCS") from exc

    bucket_path = gcs_uri[len(GCS_SCHEME):]
    bucket_name, _, prefix = bucket_path.partition("/")
    prefix = prefix.strip("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    base_path = Path(local_dir)
    print(f"Uploading {base_path} to {gcs_uri}...")
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = Path(root) / filename
            rel_path = local_path.relative_to(base_path)
            blob_path = f"{prefix}/{rel_path}".strip("/")
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
    print(f"Upload complete: {gcs_uri}")


if __name__ == "__main__":
    main()
