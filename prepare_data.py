import pandas as pd
import requests
import shutil
import os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
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


def main():
    if not os.path.exists('images'):
        os.makedirs('images')

    print("Downloading dataset.csv from HuggingFace...")
    local_path = hf_hub_download(
        repo_id="fast-stager/property-labels",
        filename="dataset.csv",
        repo_type="dataset"
    )
    shutil.copy(local_path, "dataset.csv")
    print("Downloaded dataset.csv")

    df = pd.read_csv('dataset.csv')
    print(f"dataset.csv: {len(df)} rows, {df['group_id'].nunique()} groups")
    print(f"Score distribution:\n{df['score'].value_counts().sort_index().to_string()}")

    tasks = list(zip(df.index, df['url']))

    print(f"Downloading {len(tasks)} images...")

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(download_one, tasks), total=len(tasks)))

    print("Done. Images cached in /images folder.")


if __name__ == "__main__":
    main()
