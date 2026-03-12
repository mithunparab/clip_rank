import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import asyncio

REPO_ID = "fast-stager/property-labels"
IMAGES_DIR = "images"
ASYNC_BATCH = 200
ASYNC_TIMEOUT = 10
THREAD_WORKERS = 64


def download_csv(fname):
    """Download a CSV from HuggingFace if not present locally."""
    if os.path.exists(fname):
        print(f"{fname} exists locally, skipping download")
        return
    print(f"Downloading {fname}...")
    local = hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset")
    shutil.copy(local, fname)


def collect_tasks(mode):
    """Collect (idx, url) pairs based on mode."""
    tasks = []

    if mode in ("ranking", "all"):
        download_csv("dataset.csv")
        with open("dataset.csv") as f:
            for i, row in enumerate(csv.DictReader(f)):
                tasks.append((i, row['url']))
        print(f"dataset.csv: {len(tasks)} images")

    if mode in ("binary", "all"):
        download_csv("best_image_training_data.csv")
        offset = len(tasks)
        with open("best_image_training_data.csv") as f:
            for i, row in enumerate(csv.DictReader(f)):
                tasks.append((offset + i, row['image_url']))
        print(f"best_image_training_data.csv: {len(tasks) - offset} images")

    pending = [(idx, url) for idx, url in tasks if not os.path.exists(f"{IMAGES_DIR}/{idx}.jpg")]
    print(f"Total: {len(tasks)}, cached: {len(tasks) - len(pending)}, to download: {len(pending)}")
    return pending


async def _fetch_one(session, idx, url, sem):
    import aiohttp
    path = f"{IMAGES_DIR}/{idx}.jpg"
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=ASYNC_TIMEOUT)) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    with open(path, 'wb') as f:
                        f.write(data)
                    return True
        except Exception:
            pass
    return False


async def download_async(tasks):
    import aiohttp
    sem = asyncio.Semaphore(ASYNC_BATCH)
    connector = aiohttp.TCPConnector(limit=ASYNC_BATCH, ttl_dns_cache=300, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [_fetch_one(session, idx, url, sem) for idx, url in tasks]
        results = []
        for coro in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Downloading"):
            results.append(await coro)
    print(f"Downloaded {sum(results)}/{len(tasks)} images")


def download_sync(tasks):
    import requests

    def _dl(task):
        idx, url = task
        try:
            resp = requests.get(url, timeout=ASYNC_TIMEOUT)
            if resp.status_code == 200:
                with open(f"{IMAGES_DIR}/{idx}.jpg", 'wb') as f:
                    f.write(resp.content)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as ex:
        list(tqdm(ex.map(_dl, tasks), total=len(tasks), desc="Downloading"))


def run_downloads(tasks):
    if not tasks:
        print("All images already cached.")
        return
    try:
        import aiohttp  # noqa: F401
        asyncio.run(download_async(tasks))
    except ImportError:
        print("aiohttp not installed, falling back to threaded downloads")
        download_sync(tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['binary', 'ranking', 'all'], default='all',
                        help='binary=best_image only, ranking=dataset.csv only, all=both')
    args = parser.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    tasks = collect_tasks(args.mode)
    run_downloads(tasks)
    print("Done.")


if __name__ == "__main__":
    main()
