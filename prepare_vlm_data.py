"""
Convert dataset.csv → SFT and DPO JSON files for VLM training.

Reads:  dataset.csv (columns: url, group_id, score, label)
Writes: data/sft_train.json, data/sft_val.json, data/dpo_train.json
"""

import json
import os
import random
import pandas as pd
import yaml

PROMPT_TEXT = (
    "Rate this property photo on a scale of 1 to 10 based on composition, "
    "lighting, staging, and overall appeal for a real estate listing."
)

LABEL_DESCRIPTIONS = {
    "living_room": "living room",
    "kitchen": "kitchen",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
    "outdoor": "outdoor/exterior",
    "balcony": "balcony",
    "dining_room": "dining room",
    "other": "property",
}


def raw_to_rating(score: float) -> int:
    """Map raw score from [-10, 10] → integer rating [1, 10]."""
    # Linear map: -10 → 1, 10 → 10
    rating = (score + 10) / 20 * 9 + 1
    return max(1, min(10, round(rating)))


def describe_label(label: str) -> str:
    if pd.isna(label) or not label:
        return "property"
    return LABEL_DESCRIPTIONS.get(str(label).strip().lower(), "property")


def quality_word(rating: int) -> str:
    if rating >= 8:
        return "Excellent"
    if rating >= 6:
        return "Good"
    if rating >= 4:
        return "Average"
    return "Poor"


def build_response(rating: int, label: str) -> str:
    desc = describe_label(label)
    word = quality_word(rating)
    return f"Rating: {rating}/10\n{word} {desc} photo."


def make_sft_entry(row_idx: int, rating: int, label: str) -> dict:
    image_path = f"images/{row_idx}.jpg"
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": build_response(rating, label)}
                ],
            },
        ]
    }


def make_dpo_entry(row_idx: int, chosen_rating: int, rejected_rating: int, label: str) -> dict:
    image_path = f"images/{row_idx}.jpg"
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    return {
        "prompt": prompt,
        "chosen": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": build_response(chosen_rating, label)}
                ],
            }
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": build_response(rejected_rating, label)}
                ],
            }
        ],
    }


def generate_rejected_rating(true_rating: int) -> int:
    """Generate a wrong rating with gap >= 3."""
    if true_rating >= 7:
        # Gold image → reject with low score
        return random.randint(1, min(4, true_rating - 3))
    elif true_rating <= 3:
        # Bad image → reject with high score
        return random.randint(max(7, true_rating + 3), 10)
    else:
        # Mid → shift by ±4
        if random.random() < 0.5:
            return max(1, true_rating - 4)
        else:
            return min(10, true_rating + 4)


def main():
    with open("config.yml") as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg["data"]["csv_path"]
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run prepare_data.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {df['group_id'].nunique()} groups from {csv_path}")

    # Add ratings
    df["rating"] = df["score"].apply(raw_to_rating)

    # Filter to images that exist on disk
    df["file_path"] = df.index.map(lambda x: f"images/{x}.jpg")
    df = df[df["file_path"].apply(os.path.exists)].copy()
    print(f"After filtering missing images: {len(df)} rows")

    if df.empty:
        print("No images found. Run prepare_data.py first.")
        return

    # Train/val split by group (10% groups → val, matching train_ddp.py)
    unique_groups = df["group_id"].unique()
    val_groups = set(unique_groups[: int(len(unique_groups) * 0.1)])
    train_mask = ~df["group_id"].isin(val_groups)

    train_df = df[train_mask]
    val_df = df[~train_mask]

    # --- SFT data ---
    sft_train = []
    for idx, row in train_df.iterrows():
        sft_train.append(make_sft_entry(idx, row["rating"], row.get("label", "")))

    sft_val = []
    for idx, row in val_df.iterrows():
        sft_val.append(make_sft_entry(idx, row["rating"], row.get("label", "")))

    # --- DPO data (train only) ---
    dpo_train = []
    for idx, row in train_df.iterrows():
        true_rating = row["rating"]
        rejected_rating = generate_rejected_rating(true_rating)
        dpo_train.append(make_dpo_entry(idx, true_rating, rejected_rating, row.get("label", "")))

        # Extra hard negative for gold images: rejected = true - 2
        if true_rating >= 7:
            hard_rejected = max(1, true_rating - 2)
            if hard_rejected != true_rating:
                dpo_train.append(
                    make_dpo_entry(idx, true_rating, hard_rejected, row.get("label", ""))
                )

    # Save
    os.makedirs("data", exist_ok=True)

    for path, data in [
        ("data/sft_train.json", sft_train),
        ("data/sft_val.json", sft_val),
        ("data/dpo_train.json", dpo_train),
    ]:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {path}: {len(data)} entries")

    # Print distribution
    print(f"\nRating distribution (train):")
    print(train_df["rating"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    random.seed(42)
    main()
