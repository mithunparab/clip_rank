"""
Diagnose data quality and create dataset.csv by merging:
  - verifications.csv (corrected scores, takes priority)
  - annotations.csv   (original scores, fills gaps)
Output: dataset.csv with columns [group_id, url, score, label]
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


def build_dataset(annot_path='annotations.csv', ver_path='verifications.csv',
                  out_path='dataset.csv'):
    """Merge verifications on top of annotations, dedup by URL (verifications win)."""
    print(f"\n{'='*60}")
    print("BUILDING dataset.csv")
    print(f"{'='*60}")

    annot_df = pd.read_csv(annot_path) if os.path.exists(annot_path) else None
    ver_df = pd.read_csv(ver_path) if os.path.exists(ver_path) else None

    if annot_df is None and ver_df is None:
        print("[ERROR] Neither annotations.csv nor verifications.csv found!")
        return None

    # Normalize annotations: keep [group_id, url, score, label]
    rows = []
    if annot_df is not None:
        ann = annot_df[['group_id', 'url', 'score']].copy()
        ann['label'] = annot_df['label'] if 'label' in annot_df.columns else ''
        ann['source'] = 'annotations'
        rows.append(ann)
        print(f"  annotations.csv: {len(ann)} rows, {ann['group_id'].nunique()} groups")

    # Normalize verifications: corrected_score -> score, corrected_label -> label
    if ver_df is not None:
        ver = ver_df[['group_id', 'url']].copy()
        ver['score'] = ver_df['corrected_score']
        ver['label'] = ver_df['corrected_label'] if 'corrected_label' in ver_df.columns else ''
        ver['source'] = 'verifications'
        rows.append(ver)
        print(f"  verifications.csv: {len(ver)} rows, {ver['group_id'].nunique()} groups")

    combined = pd.concat(rows, ignore_index=True)

    # Dedup by URL: verifications take priority (keep last since verifications appended second)
    before = len(combined)
    combined = combined.drop_duplicates(subset='url', keep='last')
    after = len(combined)
    print(f"  Combined: {before} -> {after} after dedup (verifications win on overlap)")

    # Count sources in final
    src_counts = combined['source'].value_counts()
    for s, c in src_counts.items():
        print(f"    from {s}: {c}")

    # Drop source column, save
    dataset = combined[['group_id', 'url', 'score', 'label']].copy()
    dataset.to_csv(out_path, index=False)
    print(f"  Saved {out_path}: {len(dataset)} rows, {dataset['group_id'].nunique()} groups")
    return dataset


def analyze_csv(df, label):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"NaN counts:\n{df.isnull().sum().to_string()}")
    if 'score' in df.columns:
        print(f"\nScore stats: min={df['score'].min()}, max={df['score'].max()}, "
              f"mean={df['score'].mean():.2f}")
        print(f"Score distribution:\n{df['score'].value_counts().sort_index().to_string()}")
    if 'label' in df.columns:
        non_null = df['label'].dropna()
        if len(non_null) > 0:
            print(f"\nLabel counts:\n{non_null.value_counts().to_string()}")
    if 'group_id' in df.columns:
        n_groups = df['group_id'].nunique()
        group_sizes = df.groupby('group_id').size()
        print(f"\nGroups: {n_groups} | avg size: {group_sizes.mean():.1f} | "
              f"min: {group_sizes.min()} | max: {group_sizes.max()}")


def analyze_tier_distribution(df, label):
    print(f"\n{'='*60}")
    print(f"TIER DISTRIBUTION — {label}")
    print(f"{'='*60}")

    df = df.copy()
    df['label'] = df['label'].fillna('')
    df['remapped'] = df.apply(lambda r: _remap_score(float(r['score']), r['label']), axis=1)
    df['tier'] = df['remapped'].apply(tier)

    print(f"Tier counts (after _remap_score):")
    for t in ['gold', 'silver', 'bronze']:
        c = (df['tier'] == t).sum()
        print(f"  {t}: {c} ({c/len(df):.1%})")

    # Per-group best tier
    group_best = df.groupby('group_id')['remapped'].max().apply(tier)
    print(f"\nBest-tier-per-group:")
    for t in ['gold', 'silver', 'bronze']:
        c = (group_best == t).sum()
        print(f"  {t}: {c} ({c/len(group_best):.1%})")

    # Single-tier groups
    diversity = df.groupby('group_id')['tier'].nunique()
    single = (diversity == 1).sum()
    total = len(diversity)
    print(f"\nSingle-tier groups (no discrimination signal): "
          f"{single}/{total} ({single/total:.1%})")

    return df


def analyze_gold_difficulty(df, label):
    print(f"\n{'='*60}")
    print(f"GOLD DIFFICULTY ANALYSIS — {label}")
    print(f"{'='*60}")

    df = df.copy()
    grouped = df.groupby('group_id')

    gold_groups, no_gold_groups = [], []
    for gid, g in grouped:
        if g['score'].max() >= 7:
            gold_groups.append(g)
        else:
            no_gold_groups.append(g)

    print(f"Groups WITH gold (score>=7): {len(gold_groups)}")
    print(f"Groups WITHOUT gold:         {len(no_gold_groups)}")

    if gold_groups:
        gold_counts, group_sizes, gold_margins = [], [], []
        for g in gold_groups:
            scores = g['score'].values
            n_gold = (scores >= 7).sum()
            gold_counts.append(n_gold)
            group_sizes.append(len(scores))
            non_gold = scores[scores < 7]
            if len(non_gold) > 0:
                gold_margins.append(scores[scores >= 7].max() - non_gold.max())
            else:
                gold_margins.append(0)

        gold_counts = np.array(gold_counts)
        group_sizes = np.array(group_sizes)
        gold_margins = np.array(gold_margins)

        print(f"\nGold group stats:")
        print(f"  Avg gold images/group: {gold_counts.mean():.1f}")
        print(f"  Only 1 gold image:     {(gold_counts == 1).sum()} ({(gold_counts == 1).mean():.1%})")
        print(f"  2+ gold images:        {(gold_counts >= 2).sum()} ({(gold_counts >= 2).mean():.1%})")
        print(f"  Avg group size:        {group_sizes.mean():.1f}")
        print(f"  Tiny groups (<=2):     {(group_sizes <= 2).sum()}")
        print(f"\n  Gold margin (best gold - best non-gold):")
        print(f"    mean={gold_margins.mean():.1f} | median={np.median(gold_margins):.1f}")
        print(f"    Hard (margin<3): {(gold_margins < 3).sum()} ({(gold_margins < 3).mean():.1%})")
        print(f"    Easy (margin>=5): {(gold_margins >= 5).sum()} ({(gold_margins >= 5).mean():.1%})")

    if no_gold_groups:
        best = pd.Series([g['score'].max() for g in no_gold_groups])
        print(f"\nNo-gold group best scores:\n  {best.value_counts().sort_index().to_string()}")


def compare_scores(annot_path='annotations.csv', ver_path='verifications.csv'):
    print(f"\n{'='*60}")
    print("SCORE CHANGES: annotations vs verifications")
    print(f"{'='*60}")

    annot = pd.read_csv(annot_path)
    ver = pd.read_csv(ver_path)

    orig = annot.drop_duplicates('url').set_index('url')['score']
    corr = ver.drop_duplicates('url').set_index('url')['corrected_score']

    common = orig.index.intersection(corr.index)
    orig_c = orig.loc[common].reset_index(drop=True)
    corr_c = corr.loc[common].reset_index(drop=True)

    changed = (orig_c != corr_c)
    n_changed = changed.sum()

    print(f"URLs in annotations:    {len(orig)}")
    print(f"URLs in verifications:  {len(corr)}")
    print(f"Common URLs:            {len(common)}")
    print(f"New in verifications:   {len(corr) - len(common)}")
    print(f"Scores CHANGED:         {n_changed}/{len(common)} ({n_changed/len(common):.1%})")

    if n_changed > 0:
        delta = corr_c[changed] - orig_c[changed]
        print(f"\nScore delta (new - old):")
        print(f"  mean={delta.mean():.2f} | median={delta.median():.1f} | "
              f"min={delta.min()} | max={delta.max()}")

        orig_tiers = orig_c[changed].apply(tier)
        new_tiers = corr_c[changed].apply(tier)
        flipped = (orig_tiers.values != new_tiers.values).sum()
        print(f"  Tier flips: {flipped}")

        flip_df = pd.crosstab(
            pd.Series(orig_tiers.values, name='original'),
            pd.Series(new_tiers.values, name='new')
        )
        print(f"\nTier transition matrix:\n{flip_df.to_string()}")


def check_val_split(df):
    print(f"\n{'='*60}")
    print("VALIDATION SPLIT PREVIEW (10%)")
    print(f"{'='*60}")

    unique_groups = df['group_id'].unique()
    n_val = int(len(unique_groups) * 0.1)
    val_groups = set(unique_groups[:n_val])

    for name, mask in [('train', ~df['group_id'].isin(val_groups)),
                       ('val', df['group_id'].isin(val_groups))]:
        split = df[mask].copy()
        split['label'] = split['label'].fillna('')
        split['remapped'] = split.apply(lambda r: _remap_score(float(r['score']), r['label']), axis=1)
        best = split.groupby('group_id')['remapped'].max()
        gold = (best >= 8).mean()
        silver = ((best >= 3) & (best < 8)).mean()
        bronze = (best < 3).mean()
        n_groups = split['group_id'].nunique()
        print(f"  {name:5s}: {n_groups:4d} groups | gold={gold:.1%} silver={silver:.1%} bronze={bronze:.1%}")


def main():
    # 1. Build dataset.csv from verifications + annotations
    dataset = build_dataset()
    if dataset is None:
        return

    # 2. Analyze merged dataset
    analyze_csv(dataset, 'MERGED dataset.csv')

    # 3. Tier distribution
    analyze_tier_distribution(dataset, 'dataset.csv')

    # 4. Gold difficulty
    analyze_gold_difficulty(dataset, 'dataset.csv')

    # 5. Score changes from original -> verified
    if os.path.exists('annotations.csv') and os.path.exists('verifications.csv'):
        compare_scores()

    # 6. Val split preview
    check_val_split(dataset)

    print(f"\n{'='*60}")
    print("DONE — dataset.csv is ready for training")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
