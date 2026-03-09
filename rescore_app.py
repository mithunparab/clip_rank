"""
Gradio app to rescore images in the 3-6 score gap.
Fills the taste manifold between 'meh' (0-2) and 'gold' (7+).

Reads from dataset.csv (never writes to it).
All rescoring goes to rescore_progress.csv.
Run merge_rescores() or the Merge button to apply to dataset.csv when done.

Usage: python rescore_app.py
"""
import gradio as gr
import pandas as pd
import numpy as np
import os
from datetime import datetime

CSV_PATH = "dataset.csv"
PROGRESS_PATH = "rescore_progress.csv"


def load_data():
    df = pd.read_csv(CSV_PATH)
    reviewed = set()
    if os.path.exists(PROGRESS_PATH) and os.path.getsize(PROGRESS_PATH) > 0:
        prog = pd.read_csv(PROGRESS_PATH)
        reviewed = set(prog['url'].values)
    return df, reviewed


def get_candidates(df, reviewed):
    """Smart filter: only score=0 images in gold groups where 0s are the minority.
    ~1,000 images instead of 5,000."""
    gold_groups = set(df[df['score'] >= 7]['group_id'])
    zeros = df[(df['score'] == 0) & df['group_id'].isin(gold_groups)].copy()

    group_stats = df[df['group_id'].isin(gold_groups)].groupby('group_id').agg(
        total=('score', 'size'),
        n_zero=('score', lambda x: (x == 0).sum())
    )
    group_stats['zero_ratio'] = group_stats['n_zero'] / group_stats['total']
    good_groups = set(group_stats[
        (group_stats['zero_ratio'] < 0.5) & (group_stats['n_zero'] <= 3)
    ].index)

    candidates = zeros[zeros['group_id'].isin(good_groups) & ~zeros['url'].isin(reviewed)]
    return candidates


def save_progress(url, old_score, new_score):
    """Append to rescore_progress.csv."""
    row = pd.DataFrame([{
        'url': url,
        'old_score': old_score,
        'new_score': new_score,
        'timestamp': datetime.now().isoformat()
    }])
    if os.path.exists(PROGRESS_PATH) and os.path.getsize(PROGRESS_PATH) > 0:
        row.to_csv(PROGRESS_PATH, mode='a', header=False, index=False)
    else:
        row.to_csv(PROGRESS_PATH, index=False)


def merge_rescores():
    """Apply rescore_progress.csv on top of dataset.csv."""
    if not os.path.exists(PROGRESS_PATH) or os.path.getsize(PROGRESS_PATH) == 0:
        return "Nothing to merge — rescore_progress.csv is empty."

    df = pd.read_csv(CSV_PATH)
    prog = pd.read_csv(PROGRESS_PATH)

    # Only apply rows where score actually changed
    changed = prog[prog['old_score'] != prog['new_score']].drop_duplicates('url', keep='last')

    score_map = dict(zip(changed['url'], changed['new_score']))
    updated = 0
    for url, new_score in score_map.items():
        mask = df['url'] == url
        if mask.any():
            df.loc[mask, 'score'] = new_score
            updated += 1

    df.to_csv(CSV_PATH, index=False)
    return f"Merged {updated} score changes into {CSV_PATH}"


def build_app():
    df, reviewed = load_data()
    candidates = get_candidates(df, reviewed)

    groups_with_candidates = candidates['group_id'].unique()
    np.random.shuffle(groups_with_candidates)

    state = {
        'df': df,
        'reviewed': reviewed,
        'group_queue': list(groups_with_candidates),
        'current_group_id': None,
        'current_group_images': [],
        'current_image_idx': 0,
        'changes': 0,
    }

    def get_stats():
        # Count gap images from progress file (not df, since we don't touch df)
        gap_in_csv = df[df['score'].between(3, 6)].shape[0]
        gap_from_progress = 0
        if os.path.exists(PROGRESS_PATH) and os.path.getsize(PROGRESS_PATH) > 0:
            prog = pd.read_csv(PROGRESS_PATH)
            gap_from_progress = prog[prog['new_score'].between(3, 6) &
                                     (prog['old_score'] != prog['new_score'])].drop_duplicates('url', keep='last').shape[0]
        total_reviewed = len(state['reviewed'])
        remaining = get_candidates(state['df'], state['reviewed']).shape[0]
        return (f"Gap images (3-6): {gap_in_csv}+{gap_from_progress} new | "
                f"Reviewed: {total_reviewed} | Remaining: {remaining} | "
                f"Changes: {state['changes']}")

    def load_next_group():
        while state['group_queue']:
            gid = state['group_queue'].pop(0)
            group = state['df'][state['df']['group_id'] == gid].copy()
            cand_urls = set(get_candidates(state['df'], state['reviewed'])['url'])
            cands = group[group['url'].isin(cand_urls)]
            if len(cands) == 0:
                continue

            state['current_group_id'] = gid
            all_imgs = []
            for _, row in group.iterrows():
                all_imgs.append({
                    'url': row['url'],
                    'score': row['score'],
                    'is_candidate': row['url'] in cands['url'].values
                })
            all_imgs.sort(key=lambda x: (-x['is_candidate'], -x['score']))
            state['current_group_images'] = all_imgs

            state['current_image_idx'] = 0
            for i, img in enumerate(all_imgs):
                if img['is_candidate']:
                    state['current_image_idx'] = i
                    break
            return True
        return False

    def get_current_display():
        if not state['current_group_images']:
            if not load_next_group():
                return None, "All done! No more candidates.", "", get_stats()

        imgs = state['current_group_images']
        idx = state['current_image_idx']
        current = imgs[idx]

        context_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
        for i, img in enumerate(imgs):
            border = "3px solid #ff6b00" if i == idx else "1px solid #444"
            label_color = "#ff6b00" if img['is_candidate'] else "#888"
            context_html += f"""
            <div style='text-align:center;'>
                <img src='{img['url']}' style='width:120px;height:90px;object-fit:cover;border:{border};border-radius:4px;'/>
                <div style='font-size:12px;color:{label_color};'>score: {img['score']}</div>
            </div>"""
        context_html += "</div>"

        return (current['url'], f"Group: {state['current_group_id']} | "
                f"Image {idx+1}/{len(imgs)} | Current score: {current['score']}",
                context_html, get_stats())

    def apply_score(new_score):
        imgs = state['current_group_images']
        idx = state['current_image_idx']
        current = imgs[idx]
        url = current['url']
        old_score = current['score']
        new_score = int(new_score)

        # Save to progress file only
        state['reviewed'].add(url)
        save_progress(url, old_score, new_score)
        if new_score != old_score:
            state['changes'] += 1

        # Move to next candidate in group
        found_next = False
        for i in range(idx + 1, len(imgs)):
            if imgs[i]['is_candidate'] and imgs[i]['url'] not in state['reviewed']:
                state['current_image_idx'] = i
                found_next = True
                break

        if not found_next:
            state['current_group_images'] = []
            if not load_next_group():
                return (None, "All done! No more candidates.", "", get_stats())

        return get_current_display()

    def skip():
        imgs = state['current_group_images']
        idx = state['current_image_idx']
        current = imgs[idx]
        return apply_score(current['score'])

    def do_merge():
        return merge_rescores()

    # UI
    with gr.Blocks(title="Rescore: Fill the 3-6 Gap") as app:
        gr.Markdown("# Rescore Images: Fill the 3-6 Gap\n"
                     "Score=0 images in gold groups. All changes saved to `rescore_progress.csv`.\n"
                     "Click **Merge into dataset.csv** when done.")

        stats_display = gr.Textbox(label="Progress", interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                main_image = gr.Image(label="Current Image", height=400)
                image_info = gr.Textbox(label="Info", interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("### Set Score")
                score_slider = gr.Slider(minimum=-10, maximum=10, step=1, value=3,
                                          label="New Score")
                with gr.Row():
                    apply_btn = gr.Button("Apply Score", variant="primary", size="lg")
                    skip_btn = gr.Button("Skip (keep current)", size="lg")
                gr.Markdown("---")
                merge_btn = gr.Button("Merge into dataset.csv", variant="secondary")
                merge_msg = gr.Textbox(label="", interactive=False)

        gr.Markdown("### Group Context (all images in this property)")
        context_gallery = gr.HTML()

        def init():
            url, info, ctx, stats = get_current_display()
            return url, info, ctx, stats, 3

        def on_apply(score):
            url, info, ctx, stats = apply_score(score)
            return url, info, ctx, stats, 3

        def on_skip():
            url, info, ctx, stats = skip()
            return url, info, ctx, stats, 3

        apply_btn.click(
            on_apply,
            inputs=[score_slider],
            outputs=[main_image, image_info, context_gallery, stats_display, score_slider]
        )
        skip_btn.click(
            on_skip,
            outputs=[main_image, image_info, context_gallery, stats_display, score_slider]
        )
        merge_btn.click(do_merge, outputs=[merge_msg])

        app.load(
            init,
            outputs=[main_image, image_info, context_gallery, stats_display, score_slider]
        )

    return app


if __name__ == "__main__":
    # Clear progress file for fresh start
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
        print(f"Cleared {PROGRESS_PATH}")

    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
