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


def merge_with_verifications(annotations_path, verifications_path):
    """
    Load annotations and apply corrected values from verifications.
    Verifications override score (and label if present) for matching URLs.
    """
    df = pd.read_csv(annotations_path)

    if not os.path.exists(verifications_path):
        print(f"No verifications file found at '{verifications_path}', using annotations only.")
        return df

    ver = pd.read_csv(verifications_path)
    print(f"Loaded {len(ver)} verification records to apply as overrides.")
    print(f"verifications.csv columns: {list(ver.columns)}")

    # Auto-detect score column
    score_col_candidates = ['score', 'corrected_score', 'verified_score', 'new_score', 'final_score']
    score_col = next((c for c in score_col_candidates if c in ver.columns), None)
    if score_col is None:
        raise ValueError(f"Cannot find score column in verifications.csv. Columns: {list(ver.columns)}")
    if score_col != 'score':
        print(f"Using '{score_col}' as the score column from verifications.csv")

    label_col_candidates = ['label', 'corrected_label', 'verified_label', 'new_label']
    label_col = next((c for c in label_col_candidates if c in ver.columns), None)

    # Index annotations by url for fast lookup
    df = df.set_index('url')

    for _, row in ver.iterrows():
        url = row['url']
        if url in df.index:
            df.at[url, 'score'] = row[score_col]
            if label_col and pd.notna(row.get(label_col)):
                df.at[url, 'label'] = row[label_col]
        else:
            # URL only in verifications: add as new row
            new_row = row.reindex(df.columns)
            df.loc[url] = new_row

    df = df.reset_index()
    print(f"Merged dataset: {len(df)} total records.")
    return df


def main():
    if not os.path.exists('images'):
        os.makedirs('images')

    df = merge_with_verifications('dataset.csv', 'verifications.csv')

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
