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
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import PropertyPreferenceDataset, _remap_score
from model import MobileCLIPRanker
from utils import load_config


THUMB_W, THUMB_H = 420, 315
CAPTION_H = 60
GAP = 20
PAGE_PAD = 20


def _thumb(path_or_url):
    """Load a local image as an RGB PIL thumbnail. Returns placeholder on failure."""
    img = None
    try:
        if path_or_url and os.path.exists(path_or_url):
            img = Image.open(path_or_url).convert('RGB')
    except Exception:
        img = None
    if img is None:
        img = Image.new('RGB', (THUMB_W, THUMB_H), (60, 60, 60))
    img = img.copy()
    img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
    # Pad to uniform size
    canvas = Image.new('RGB', (THUMB_W, THUMB_H), (20, 20, 20))
    canvas.paste(img, ((THUMB_W - img.width) // 2, (THUMB_H - img.height) // 2))
    return canvas


def _load_font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
             '/System/Library/Fonts/Helvetica.ttc'):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/best_model.pth')
    parser.add_argument('--config', default='config.yml')
    parser.add_argument('--n', type=int, default=30, help='number of misses to show')
    parser.add_argument('--out', default='misses.jpg',
                        help='Output image file (jpg/png). Viewable in notebook or image viewer.')
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
    print(f"Rendering top {len(misses)} confident misses (of ~{total} val groups) → {args.out}")

    if not misses:
        print("No misses found — either the model is perfect or val is empty.")
        return

    # Two thumbnails side by side + caption below. Pack rows into a single tall image.
    row_w = THUMB_W * 2 + GAP * 3
    row_h = THUMB_H + CAPTION_H + GAP
    img_w = row_w + PAGE_PAD * 2
    img_h = PAGE_PAD * 2 + row_h * len(misses)

    canvas = Image.new('RGB', (img_w, img_h), (17, 17, 17))
    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(16)
    font_meta = _load_font(13)

    y = PAGE_PAD
    for i, m in enumerate(misses):
        mp, gp = m['model_pick'], m['gt_pick']
        x_left = PAGE_PAD + GAP
        x_right = x_left + THUMB_W + GAP

        canvas.paste(_thumb(mp['file_path']), (x_left, y))
        canvas.paste(_thumb(gp['file_path']), (x_right, y))

        cap_y = y + THUMB_H + 6
        ties = f"  (1 of {m['n_tied_top']} tied top)" if m['n_tied_top'] > 1 else ''
        draw.text((x_left, cap_y),
                  f"#{i+1}  group {m['group_id']}  gap +{m['gap']:.2f}",
                  fill=(150, 150, 150), font=font_meta)
        draw.text((x_left, cap_y + 18),
                  f"MODEL'S PICK (wrong)  label={mp.get('label', '-') or '-'}  GT={m['model_gt']:.1f}  pred={m['model_pred']:.2f}",
                  fill=(255, 112, 112), font=font_title)
        draw.text((x_right, cap_y),
                  f"GT TOP PICK{ties}",
                  fill=(150, 150, 150), font=font_meta)
        draw.text((x_right, cap_y + 18),
                  f"label={gp.get('label', '-') or '-'}  GT={m['gt_gt']:.1f}  pred={m['gt_pred']:.2f}",
                  fill=(112, 255, 112), font=font_title)
        y += row_h

    # JPEG for size; PNG if caller chose .png
    ext = os.path.splitext(args.out)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        canvas.save(args.out, 'JPEG', quality=85, optimize=True)
    else:
        canvas.save(args.out)
    print(f"Saved {args.out} ({img_w}x{img_h} px).")
    print("In a Kaggle notebook:  from IPython.display import Image; Image(filename='" + args.out + "')")


if __name__ == '__main__':
    main()
