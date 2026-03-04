"""
VLM-based property image ranker.

Mirrors the PropertyRanker interface from inference.py but uses a
fine-tuned Qwen VLM that generates "Rating: X/10" text scores.
"""

import os
import re
import time
import yaml
import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor


PROMPT_TEXT = (
    "Rate this property photo on a scale of 1 to 10 based on composition, "
    "lighting, staging, and overall appeal for a real estate listing."
)


class VLMPropertyRanker:
    def __init__(self, checkpoint_path=None, config_path="config.yml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        vlm_cfg = cfg.get("vlm", {})

        # Find checkpoint: explicit > DPO > SFT
        if checkpoint_path is None:
            dpo_dir = vlm_cfg.get("dpo", {}).get("save_dir", "checkpoints_vlm_dpo")
            sft_dir = vlm_cfg.get("sft", {}).get("save_dir", "checkpoints_vlm_sft")
            if os.path.isdir(f"{dpo_dir}/final"):
                checkpoint_path = f"{dpo_dir}/final"
                print(f"Using DPO checkpoint: {checkpoint_path}")
            elif os.path.isdir(f"{sft_dir}/final"):
                checkpoint_path = f"{sft_dir}/final"
                print(f"Using SFT checkpoint (DPO not found): {checkpoint_path}")
            else:
                raise FileNotFoundError(
                    f"No VLM checkpoint found in {dpo_dir}/final or {sft_dir}/final"
                )

        print(f"Loading VLM from {checkpoint_path}...")
        from unsloth import FastVisionModel

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            checkpoint_path,
            load_in_4bit=vlm_cfg.get("load_in_4bit", False),
        )
        FastVisionModel.for_inference(self.model)

        # Load the processor for handling images + text together
        base_model = vlm_cfg.get("base_model", "unsloth/Qwen3.5-0.8B")
        self.processor = AutoProcessor.from_pretrained(base_model)
        print("VLM loaded successfully.")

    def _load_image(self, src: str) -> Image.Image:
        if src.startswith("http"):
            resp = requests.get(src, timeout=10)
            return Image.open(BytesIO(resp.content)).convert("RGB")
        return Image.open(src).convert("RGB")

    def _parse_rating(self, text: str) -> float:
        """Extract rating from generated text. Multiple fallback patterns."""
        # Pattern 1: "Rating: X/10"
        m = re.search(r"Rating:\s*(\d+(?:\.\d+)?)\s*/\s*10", text)
        if m:
            return float(m.group(1))

        # Pattern 2: any "X/10"
        m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
        if m:
            return float(m.group(1))

        # Pattern 3: standalone number 1-10
        m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 10:
                return val

        return 5.0  # default fallback

    def _score_image(self, image: Image.Image) -> tuple:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            }
        ]

        # Use tokenizer for chat template, processor for image+text encoding
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[input_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        # Decode only new tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, prompt_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        score = self._parse_rating(response)
        return score, response

    def rank(self, image_list: list) -> list:
        """Score and rank images. Returns list of dicts sorted by score descending."""
        results = []

        print(f"Processing {len(image_list)} images...")
        for src in image_list:
            if not src or not isinstance(src, str):
                continue
            try:
                img = self._load_image(src)
                score, response = self._score_image(img)
                results.append({
                    "source": src,
                    "score": score,
                    "response": response.strip(),
                })
            except Exception as e:
                print(f"Error processing {src}: {e}")

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


if __name__ == "__main__":
    ranker = VLMPropertyRanker()

    test_urls = [
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m3865337706s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m1211374265s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m713883090s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/c3065cb0efd74e0e69c634c4e7926ed0l-m3456441259s-w2048_h1536.jpg",
    ]

    start_time = time.time()
    results = ranker.rank(test_urls)
    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("VLM RANKING RESULTS")
    print("=" * 50)

    for i, res in enumerate(results):
        print(f"{i+1}. Score: {res['score']:.1f}/10 | {res['source']}")
        print(f"   Response: {res['response'][:80]}")

    print("=" * 50)
    print(f"Total Time: {elapsed:.2f}s")
    print("=" * 50)
