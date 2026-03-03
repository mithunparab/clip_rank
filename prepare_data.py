import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def download_one(row):
    idx, url = row
    filename = f"images/{idx}.jpg"

    if os.path.exists(filename):
        return

    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(resp.content)
    except:
        pass


def build_dataset(annotations_path, verifications_path):
    """
    Use verifications.csv as the primary dataset (human-reviewed, [-10, 10] scale).
    Join with original annotations to recover `label` where verifications lack it.
    Falls back to annotations alone if verifications.csv is missing.
    """
    ann = pd.read_csv(annotations_path)

    if not os.path.exists(verifications_path):
        print(f"No verifications file found — using annotations only.")
        return ann

    ver = pd.read_csv(verifications_path)
    print(f"verifications.csv: {len(ver)} rows, {ver['group_id'].nunique()} groups")
    print(f"verifications.csv columns: {list(ver.columns)}")

    # verifications uses corrected_score ([-10, 10] scale) — rename to 'score'
    ver = ver.rename(columns={'corrected_score': 'score'})

    # Pull labels from original annotations since verifications.corrected_label is all NaN
    url_to_label = ann.set_index('url')['label'].dropna()
    ver['label'] = ver['url'].map(url_to_label)

    df = ver[['url', 'group_id', 'score', 'label']].copy()

    score_counts = df['score'].value_counts().sort_index()
    print(f"Score distribution:\n{score_counts.to_string()}")
    print(f"Label NaN: {df['label'].isna().sum()} / {len(df)}")
    print(f"Total: {len(df)} rows, {df['group_id'].nunique()} groups")

    return df


def main():
    if not os.path.exists('images'):
        os.makedirs('images')

    df = build_dataset('dataset.csv', 'verifications.csv')

    # Save merged result back so training uses corrected scores
    df.to_csv('dataset.csv', index=False)
    print("Saved merged dataset to dataset.csv")

    tasks = list(zip(df.index, df['url']))

    print(f"Downloading {len(tasks)} images...")

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(download_one, tasks), total=len(tasks)))

    print("Done. Images cached in /images folder.")


if __name__ == "__main__":
    main()
