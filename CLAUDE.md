# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Property image ranking system using Apple's MobileCLIP visual encoder. A fine-tuned MobileCLIP backbone produces per-image ranking scores via a linear head, trained with a listwise KL-divergence loss on grouped property photos.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Data pipeline
python download.py         # Downloads annotations.csv from HuggingFace (fast-stager/property-labels)
python prepare_data.py     # Downloads images via ThreadPoolExecutor into images/

# Training (supports DDP for multi-GPU)
python train_ddp.py

# Inference
python inference.py
```

No test framework is configured. Validation runs during training via strict accuracy metrics.

## Architecture

**Model (`model.py` - MobileCLIPRanker):** Wraps a MobileCLIP visual encoder (default: L14 variant) as a frozen backbone with only the last 60 parameters unfrozen. A single linear head maps backbone embeddings to scalar ranking scores. Input shape: `(batch, group, C, H, W)`.

**Dataset (`dataset.py` - PropertyPreferenceDataset):** Groups images by `group_id` (padded to 15 per group). Scoring rules remap labels: outdoor/bathroom/balcony with score>=8 → 0.0, bedroom with score>=8 → 3.0. Uses CLIP-specific normalization (mean=0.481/0.457/0.408, std=0.268/0.261/0.275).

**Training (`train_ddp.py`):** Listwise KL-divergence loss over score tiers (gold>=8, silver>=3). AdamW with differential learning rates (head: 1e-3, backbone: 1e-5). Cosine annealing scheduler. Early stopping with patience=15. Checkpoints saved to `config.train.save_dir`.

**Inference (`inference.py` - PropertyRanker):** Loads a trained checkpoint, accepts HTTP URLs or local paths, returns images sorted by predicted score.

## Configuration

All hyperparameters and paths are in `config.yml`. Key settings: model variant, learning rates, batch size, image size (224), max images per group (15), cross-validation folds.
