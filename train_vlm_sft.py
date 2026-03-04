"""
Phase 1: Supervised Fine-Tuning of Qwen VLM for property photo scoring.

Teaches the model to output "Rating: X/10" for property images.
Uses unsloth FastVisionModel + LoRA for efficient training.
"""

import json
import yaml
from types import SimpleNamespace
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from datasets import Dataset


def load_vlm_config(path="config.yml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("vlm", {})


def convert_to_hf_messages(entry):
    """Convert our JSON format to HF conversation format."""
    messages = []
    for msg in entry["messages"]:
        if msg["role"] == "user":
            content_parts = []
            for part in msg["content"]:
                if part["type"] == "image":
                    content_parts.append({"type": "image", "image": part["image"]})
                elif part["type"] == "text":
                    content_parts.append({"type": "text", "text": part["text"]})
            messages.append({"role": "user", "content": content_parts})
        elif msg["role"] == "assistant":
            text = msg["content"][0]["text"]
            messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return {"messages": messages}


def main():
    vlm_cfg = load_vlm_config()
    lora_cfg = vlm_cfg.get("lora", {})
    sft_cfg = vlm_cfg.get("sft", {})

    base_model = vlm_cfg.get("base_model", "unsloth/Qwen3.5-0.8B")
    load_in_4bit = vlm_cfg.get("load_in_4bit", False)

    print(f"Loading model: {base_model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0),
        bias="none",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Load SFT data
    with open("data/sft_train.json") as f:
        train_data = json.load(f)
    with open("data/sft_val.json") as f:
        val_data = json.load(f)

    train_entries = [convert_to_hf_messages(e) for e in train_data]
    val_entries = [convert_to_hf_messages(e) for e in val_data]

    train_dataset = Dataset.from_list(train_entries)
    val_dataset = Dataset.from_list(val_entries)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    save_dir = sft_cfg.get("save_dir", "checkpoints_vlm_sft")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=sft_cfg.get("batch_size", 2),
            gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
            warmup_steps=sft_cfg.get("warmup_steps", 50),
            max_steps=sft_cfg.get("max_steps", 500),
            learning_rate=sft_cfg.get("lr", 2e-4),
            fp16=not FastVisionModel.supports_bf16(),
            bf16=FastVisionModel.supports_bf16(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=sft_cfg.get("eval_steps", 50),
            save_strategy="steps",
            save_steps=sft_cfg.get("save_steps", 100),
            optim="adamw_8bit",
            weight_decay=sft_cfg.get("weight_decay", 0.01),
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=save_dir,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
        ),
    )

    print("Starting SFT training...")
    trainer.train()

    # Save final adapter
    final_dir = f"{save_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"SFT adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
