"""
Diagnose why model accuracy dropped from 97.72% to ~80%.
Run this on Kaggle BEFORE training to understand the data.
"""
import pandas as pd
import numpy as np
import os


def _remap_score(raw, label):
    lbl = str(label).lower()
    if raw >= 8:
        if lbl in ['outdoor', 'bathroom', 'other', 'balcony']:
            return 0.0
        if lbl == 'bedroom':
            return 3.0
    return raw


def tier(score):
    if score >= 8:
        return 'gold'
    elif score >= 3:
        return 'silver'
    return 'bronze'


def analyze_csv(path, label):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"{label}: {path}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"NaN counts:\n{df.isnull().sum().to_string()}")
    if 'score' in df.columns:
        print(f"\nScore stats: min={df['score'].min()}, max={df['score'].max()}, "
              f"mean={df['score'].mean():.2f}")
        print(f"Score value counts:\n{df['score'].value_counts().sort_index().to_string()}")
    if 'label' in df.columns:
        print(f"\nLabel counts:\n{df['label'].value_counts().to_string()}")
    if 'group_id' in df.columns:
        n_groups = df['group_id'].nunique()
        group_sizes = df.groupby('group_id').size()
        print(f"\nGroups: {n_groups} | avg size: {group_sizes.mean():.1f} | "
              f"min: {group_sizes.min()} | max: {group_sizes.max()}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    return df


def analyze_tier_distribution(df, label):
    print(f"\n{'='*60}")
    print(f"TIER DISTRIBUTION — {label}")
    print(f"{'='*60}")

    df = df.copy()
    df['remapped'] = df.apply(lambda r: _remap_score(float(r['score']), r.get('label', '')), axis=1)
    df['tier'] = df['remapped'].apply(tier)

    print(f"Tier counts (after _remap_score):\n{df['tier'].value_counts().to_string()}")

    # Per-group: what is the best tier available?
    group_best = df.groupby('group_id').apply(
        lambda g: tier(max(g['remapped']))
    )
    print(f"\nBest-tier-per-group distribution:\n{group_best.value_counts().to_string()}")

    # Groups where ALL images are same tier (no discrimination possible)
    def group_tier_diversity(g):
        tiers = g['remapped'].apply(tier).unique()
        return len(tiers)

    diversity = df.groupby('group_id').apply(group_tier_diversity)
    single_tier_groups = (diversity == 1).sum()
    total = len(diversity)
    print(f"\nGroups with only 1 tier (trivially easy/hard): "
          f"{single_tier_groups}/{total} ({single_tier_groups/total:.1%})")

    return df


def compare_scores(orig_df, merged_df):
    print(f"\n{'='*60}")
    print("SCORE CHANGES: original vs merged")
    print(f"{'='*60}")

    orig_indexed = orig_df.set_index('url')['score']
    merged_indexed = merged_df.set_index('url')['score']

    common_urls = orig_indexed.index.intersection(merged_indexed.index)
    changed = (orig_indexed[common_urls] != merged_indexed[common_urls])
    n_changed = changed.sum()
    print(f"URLs in original: {len(orig_indexed)}")
    print(f"URLs in merged:   {len(merged_indexed)}")
    print(f"Common URLs:      {len(common_urls)}")
    print(f"New URLs (only in merged): {len(merged_indexed) - len(common_urls)}")
    print(f"Scores CHANGED by verifications: {n_changed} / {len(common_urls)} "
          f"({n_changed/len(common_urls):.1%})")

    if n_changed > 0:
        orig_changed = orig_indexed[common_urls][changed]
        new_changed = merged_indexed[common_urls][changed]
        delta = new_changed - orig_changed
        print(f"\nScore delta stats (new - original):")
        print(f"  mean: {delta.mean():.2f}, median: {delta.median():.1f}, "
              f"min: {delta.min()}, max: {delta.max()}")

        # Tier flips
        orig_tiers = orig_changed.apply(tier)
        new_tiers = new_changed.apply(tier)
        flipped = (orig_tiers != new_tiers).sum()
        print(f"  Tier flips (gold↔silver↔bronze): {flipped}")

        flip_breakdown = pd.crosstab(
            orig_tiers.rename('original_tier'),
            new_tiers.rename('new_tier')
        )
        print(f"\nTier flip breakdown:\n{flip_breakdown.to_string()}")


def check_val_split(merged_df):
    print(f"\n{'='*60}")
    print("VALIDATION SPLIT ANALYSIS")
    print(f"{'='*60}")

    unique_groups = merged_df['group_id'].unique()
    n_val = int(len(unique_groups) * 0.1)
    val_groups = set(unique_groups[:n_val])
    train_groups = set(unique_groups[n_val:])

    val_df = merged_df[merged_df['group_id'].isin(val_groups)]
    train_df = merged_df[merged_df['group_id'].isin(train_groups)]

    print(f"Val groups: {len(val_groups)} | Train groups: {len(train_groups)}")

    for split_name, split_df in [('val', val_df), ('train', train_df)]:
        split_df = split_df.copy()
        split_df['remapped'] = split_df.apply(
            lambda r: _remap_score(float(r['score']), r.get('label', '')), axis=1
        )
        group_best = split_df.groupby('group_id')['remapped'].max()
        gold_pct = (group_best >= 8).mean()
        silver_pct = ((group_best >= 3) & (group_best < 8)).mean()
        bronze_pct = (group_best < 3).mean()
        print(f"\n  {split_name}: gold-winner={gold_pct:.1%}  "
              f"silver-winner={silver_pct:.1%}  bronze-winner={bronze_pct:.1%}")


def main():
    # 1. Analyze raw files
    orig_df = analyze_csv('dataset.csv', 'CURRENT dataset.csv (post-merge if already run)')
    ver_df = analyze_csv('verifications.csv', 'verifications.csv')

    # 2. Check if annotations.csv (pre-merge) exists
    annot_df = analyze_csv('annotations.csv', 'annotations.csv (pre-merge original)')

    # 3. Tier distribution on current dataset
    if orig_df is not None and 'score' in orig_df.columns and 'label' in orig_df.columns:
        analyze_tier_distribution(orig_df, 'current dataset.csv')

    # 4. Compare original vs merged (only if both exist)
    if annot_df is not None and orig_df is not None:
        compare_scores(annot_df, orig_df)
    elif ver_df is not None and orig_df is not None:
        # verifications vs current
        print("\n[annotations.csv not found — comparing verifications vs current dataset.csv]")

    # 5. Val split composition
    if orig_df is not None and 'group_id' in orig_df.columns:
        check_val_split(orig_df)

    print("\n" + "="*60)
    print("DONE — paste the full output above for diagnosis")
    print("="*60)


if __name__ == "__main__":
    main()
