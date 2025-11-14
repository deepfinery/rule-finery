import os, json
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

import importlib.util
spec = importlib.util.spec_from_file_location("map", os.path.join(os.path.dirname(__file__), "map_facts_to_text.py"))
mapmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mapmod)

def to_chat(example):
    ex = mapmod.row_to_example(example)
    return {"text": ex["prompt"] + "\n" + ex["response"]}

def main():
    model_id = os.environ.get("MODEL_ID","meta-llama/Llama-3.2-3B-Instruct")
    train_path = os.environ.get("TRAIN_FILE","../data/train.jsonl")
    val_path   = os.environ.get("VAL_FILE","../data/val.jsonl")
    out_dir    = os.environ.get("OUT_DIR","./llama3-3b-aml-qlora")

    ds_train = load_dataset("json", data_files=train_path, split="train").map(to_chat, remove_columns=["facts","decision","fired_rules"])
    ds_val   = load_dataset("json", data_files=val_path,   split="train").map(to_chat, remove_columns=["facts","decision","fired_rules"])

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.eos_token

    lora = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )

    cfg = SFTConfig(
        output_dir=out_dir,
        max_seq_length=1024,
        packing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model_id,
        tokenizer=tok,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=lora,
        packing=True,
        args=cfg,
        dataset_text_field="text",
        quantization_config=bnb
    )
    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
