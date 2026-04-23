import argparse
import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def _download(args):
    path, url = args
    if os.path.exists(path):
        return
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            with open(path, 'wb') as f:
                f.write(resp.content)
    except Exception:
        pass


def build_dataset(annotations_path, verifications_path):
    """
    Use verifications.csv as the primary dataset (human-reviewed, [-10, 10] scale).
    Join with original annotations to recover `label` where verifications lack it.
    Falls back to annotations alone if verifications.csv is missing.
    """
    ann_label_path = 'annotations.csv'
    ann = pd.read_csv(ann_label_path if os.path.exists(ann_label_path) else annotations_path)

    if not os.path.exists(verifications_path):
        print(f"No verifications file found — using annotations only.")
        return ann

    ver = pd.read_csv(verifications_path)
    print(f"verifications.csv: {len(ver)} rows, {ver['group_id'].nunique()} groups")
    print(f"verifications.csv columns: {list(ver.columns)}")

    ver = ver.rename(columns={'corrected_score': 'score'})

    url_to_label = ann.set_index('url')['label'].dropna()
    ver['label'] = ver['url'].map(url_to_label)

    df = ver[['url', 'group_id', 'score', 'label']].copy()

    score_counts = df['score'].value_counts().sort_index()
    print(f"Score distribution:\n{score_counts.to_string()}")
    print(f"Label NaN: {df['label'].isna().sum()} / {len(df)}")
    print(f"Total: {len(df)} rows, {df['group_id'].nunique()} groups")

    return df


def run_scored():
    if not os.path.exists('images'):
        os.makedirs('images')

    df = build_dataset('annotations.csv', 'verifications.csv')
    df.to_csv('dataset.csv', index=False)
    print("Saved merged dataset to dataset.csv")

    tasks = [(f"images/{idx}.jpg", url) for idx, url in zip(df.index, df['url'])]
    print(f"Downloading {len(tasks)} images into images/ ...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(_download, tasks), total=len(tasks)))
    print("Done. Images cached in images/.")


def run_binary(csv_path, images_dir):
    """Download the binary one-hot preference set to a SEPARATE folder.

    The binary CSV is keyed by `image_id`, whose integer values can collide
    with scored-set row indexes used as filenames in images/. Keep them split.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Upload best_image_training_data.csv to the working dir "
            "(on Kaggle: add it as a dataset and symlink/copy into the working dir)."
        )
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    df = pd.read_csv(csv_path)
    print(f"{csv_path}: {len(df)} rows, columns={list(df.columns)}")
    n_groups = df['property_id'].nunique() if 'property_id' in df.columns else None
    n_selected = int(df['selected'].sum()) if 'selected' in df.columns else None
    print(f"  properties={n_groups}  selected={n_selected}")

    tasks = [
        (os.path.join(images_dir, f"{img_id}.jpg"), url)
        for img_id, url in zip(df['image_id'], df['image_url'])
    ]
    print(f"Downloading {len(tasks)} binary images into {images_dir}/ ...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(_download, tasks), total=len(tasks)))
    print(f"Done. Images cached in {images_dir}/.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['scored', 'binary'], default='scored',
                        help="scored: download verifications-scored set into images/. "
                             "binary: download best_image_training_data.csv into a separate dir.")
    parser.add_argument('--binary-csv', default='best_image_training_data.csv')
    parser.add_argument('--binary-images-dir', default='images_binary')
    args = parser.parse_args()

    if args.mode == 'scored':
        run_scored()
    else:
        run_binary(args.binary_csv, args.binary_images_dir)


if __name__ == "__main__":
    main()
