"""
Phase 2: DPO training on top of SFT checkpoint.

Sharpens preference alignment — teaches the model to prefer accurate scores
over inaccurate ones for property photos.
"""

import json
import yaml
import torch
from unsloth import FastVisionModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


def load_vlm_config(path="config.yml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("vlm", {})


def format_dpo_entry(entry):
    """Convert our JSON format to HF DPO format."""
    prompt = entry["prompt"]
    chosen = entry["chosen"]
    rejected = entry["rejected"]

    # Flatten to conversation format
    prompt_msgs = []
    for msg in prompt:
        content_parts = []
        for part in msg["content"]:
            if part["type"] == "image":
                content_parts.append({"type": "image", "image": part["image"]})
            elif part["type"] == "text":
                content_parts.append({"type": "text", "text": part["text"]})
        prompt_msgs.append({"role": msg["role"], "content": content_parts})

    chosen_text = chosen[0]["content"][0]["text"]
    rejected_text = rejected[0]["content"][0]["text"]

    return {
        "prompt": prompt_msgs,
        "chosen": [{"role": "assistant", "content": [{"type": "text", "text": chosen_text}]}],
        "rejected": [{"role": "assistant", "content": [{"type": "text", "text": rejected_text}]}],
    }


def main():
    vlm_cfg = load_vlm_config()
    dpo_cfg = vlm_cfg.get("dpo", {})
    sft_dir = vlm_cfg.get("sft", {}).get("save_dir", "checkpoints_vlm_sft")
    sft_checkpoint = f"{sft_dir}/final"

    base_model = vlm_cfg.get("base_model", "unsloth/Qwen3.5-0.8B")
    load_in_4bit = vlm_cfg.get("load_in_4bit", False)

    print(f"Loading SFT checkpoint from {sft_checkpoint}")
    model, tokenizer = FastVisionModel.from_pretrained(
        sft_checkpoint,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    # SFT checkpoint already has LoRA adapters — just enable training.
    # ref_model=None in DPOTrainer uses implicit LoRA reference.

    # Load DPO data
    with open("data/dpo_train.json") as f:
        dpo_data = json.load(f)

    entries = [format_dpo_entry(e) for e in dpo_data]
    dataset = Dataset.from_list(entries)
    print(f"DPO dataset: {len(dataset)} pairs")

    save_dir = dpo_cfg.get("save_dir", "checkpoints_vlm_dpo")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Implicit reference via LoRA
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            per_device_train_batch_size=dpo_cfg.get("batch_size", 1),
            gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 8),
            warmup_steps=dpo_cfg.get("warmup_steps", 20),
            max_steps=dpo_cfg.get("max_steps", 200),
            learning_rate=dpo_cfg.get("lr", 5e-5),
            beta=dpo_cfg.get("beta", 0.1),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            save_strategy="steps",
            save_steps=dpo_cfg.get("save_steps", 50),
            optim="adamw_8bit",
            weight_decay=dpo_cfg.get("weight_decay", 0.01),
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=save_dir,
            report_to="none",
            remove_unused_columns=False,
            max_length=2048,
            max_prompt_length=1024,
        ),
    )

    print("Starting DPO training...")
    trainer.train()

    # Save final adapter
    final_dir = f"{save_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"DPO adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
