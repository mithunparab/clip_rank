"""Inspect val-set top-1 misses from a trained ranker.

Generates an HTML grid of (model #1 pick vs GT #1 pick) for each group
where the model got top-1 wrong, sorted by how confidently wrong the
model was. Open the HTML in a browser to eyeball why the model misses —
if the model's pick is defensibly "also a good image", the 60% ceiling
is annotator subjectivity, not a model gap.

Usage:
    python inspect_misses.py --model checkpoints/best_model.pth --n 30
"""

import os
import argparse
import html as html_lib
import numpy as np
import pandas as pd
import torch

from dataset import PropertyPreferenceDataset, _remap_score
from model import MobileCLIPRanker
from utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/best_model.pth')
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--n', type=int, default=30, help='number of misses to show')
    parser.add_argument('--out', default='misses.html')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reconstruct the same val split train_regression.py uses.
    df = pd.read_csv(cfg.data.csv_path)
    if 'file_path' not in df.columns:
        df['file_path'] = df.index.map(lambda x: os.path.join('images', f"{x}.jpg"))
    seed = getattr(cfg.train, 'seed', 42)
    unique_groups = df['group_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    val_groups = unique_groups[:int(len(unique_groups) * 0.1)]
    val_df = df[df['group_id'].isin(val_groups)].copy()

    # Auto-detect head type from checkpoint keys (matches inference.py behavior).
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    state_dict = ckpt.get('model_state_dict', ckpt)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    cfg.model.use_attention = any('head.attn.' in k for k in state_dict)

    head_kind = 'attention' if cfg.model.use_attention else 'independent'
    print(f"Loading {args.model} ({head_kind} head) on {device}")
    model = MobileCLIPRanker(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ds = PropertyPreferenceDataset(
        pd.DataFrame({'group_id': [], 'score': [], 'label': []}),
        images_dir='images', is_train=False, img_size=cfg.data.img_size,
    )

    misses = []
    with torch.no_grad():
        for gid, group in val_df.groupby('group_id'):
            rows = [r for r in group.to_dict('records') if os.path.exists(r['file_path'])]
            if len(rows) < 2:
                continue

            tensors = [ds._process(r['file_path']) for r in rows]
            batch = torch.stack(tensors).unsqueeze(0).to(device)
            vlen = torch.tensor([len(tensors)])
            preds = model(batch, valid_lens=vlen).view(-1).cpu().numpy()
            gts = np.array([_remap_score(float(r['score']), r.get('label', '')) for r in rows])

            model_argmax = int(np.argmax(preds))
            gt_max = gts.max()
            best_mask = gts == gt_max
            if best_mask[model_argmax]:
                continue  # correct

            # Among tied GT-max images, pick the one the model liked most —
            # the "best shot the model had at being right".
            gt_top_idx = int(np.argmax(np.where(best_mask, preds, -np.inf)))
            misses.append({
                'group_id': gid,
                'gap': float(preds[model_argmax] - preds[gt_top_idx]),
                'model_pick': rows[model_argmax],
                'model_pred': float(preds[model_argmax]),
                'model_gt': float(gts[model_argmax]),
                'gt_pick': rows[gt_top_idx],
                'gt_pred': float(preds[gt_top_idx]),
                'gt_gt': float(gts[gt_top_idx]),
                'n_tied_top': int(best_mask.sum()),
            })

    misses.sort(key=lambda m: m['gap'], reverse=True)
    misses = misses[:args.n]
    total = sum(1 for _ in val_df.groupby('group_id'))
    print(f"Showing top {len(misses)} confident misses (of ~{total} val groups) in {args.out}")

    parts = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8"><title>Val-set misses</title>',
        '<style>',
        'body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:20px;max-width:1400px;}',
        '.row{display:flex;gap:20px;margin:24px 0;padding:16px;background:#1c1c1c;border-radius:8px;}',
        '.cell{flex:1;min-width:0;}',
        '.cell img{width:100%;height:auto;border-radius:4px;display:block;}',
        '.cell h3{margin:0 0 6px 0;font-size:14px;}',
        '.wrong h3{color:#ff7070;}',
        '.right h3{color:#70ff70;}',
        '.meta{font-size:12px;color:#aaa;margin-top:6px;line-height:1.5;}',
        '.header{font-size:12px;color:#888;margin-bottom:6px;}',
        'h1{font-size:20px;} p{color:#bbb;}',
        '</style>',
        '</head><body>',
        f'<h1>Top {len(misses)} val misses — most confidently wrong first</h1>',
        '<p>Left = model\'s #1 pick (wrong). Right = a GT-max image (what the model should\'ve picked). '
        'If ≥30% of these look like defensible calls, the ~60% ceiling is annotator subjectivity. '
        'If the model\'s picks are clearly worse, there is room to improve with model/training changes.</p>',
    ]

    for i, m in enumerate(misses):
        gid = html_lib.escape(str(m['group_id']))
        mp = m['model_pick']
        gp = m['gt_pick']
        model_src = html_lib.escape(mp.get('url') or mp['file_path'])
        gt_src = html_lib.escape(gp.get('url') or gp['file_path'])
        model_label = html_lib.escape(str(mp.get('label', '') or ''))
        gt_label = html_lib.escape(str(gp.get('label', '') or ''))
        ties_note = f' (1 of {m["n_tied_top"]} tied at top)' if m['n_tied_top'] > 1 else ''

        parts.append(
            f'<div class="row">'
            f'<div class="cell wrong">'
            f'<div class="header">#{i+1} · group {gid} · gap +{m["gap"]:.2f}</div>'
            f'<h3>Model\'s #1 pick (wrong)</h3>'
            f'<img src="{model_src}" loading="lazy">'
            f'<div class="meta">label: {model_label or "—"}<br>'
            f'GT score: {m["model_gt"]:.1f} · pred: {m["model_pred"]:.2f}</div>'
            f'</div>'
            f'<div class="cell right">'
            f'<div class="header">&nbsp;</div>'
            f'<h3>GT top pick{ties_note}</h3>'
            f'<img src="{gt_src}" loading="lazy">'
            f'<div class="meta">label: {gt_label or "—"}<br>'
            f'GT score: {m["gt_gt"]:.1f} · pred: {m["gt_pred"]:.2f}</div>'
            f'</div>'
            f'</div>'
        )

    parts.append('</body></html>')
    with open(args.out, 'w') as f:
        f.write('\n'.join(parts))
    print(f"Open {args.out} in a browser.")


if __name__ == '__main__':
    main()
