import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import aiohttp
import asyncio

REPO_ID = "fast-stager/property-labels"
IMAGES_DIR = "images"
ASYNC_BATCH = 200       # concurrent connections per batch
ASYNC_TIMEOUT = 10
THREAD_WORKERS = 64      # fallback threadpool size


def download_csvs():
    """Download both CSVs from HuggingFace."""
    for fname in ("dataset.csv", "best_image_training_data.csv"):
        print(f"Downloading {fname}...")
        local = hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="dataset")
        shutil.copy(local, fname)


def collect_tasks():
    """Read both CSVs and return list of (idx, url) for images not yet downloaded."""
    tasks = []

    with open("dataset.csv") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tasks.append((i, row['url']))
    n_rank = len(tasks)

    with open("best_image_training_data.csv") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tasks.append((n_rank + i, row['image_url']))

    # Skip already downloaded
    pending = [(idx, url) for idx, url in tasks if not os.path.exists(f"{IMAGES_DIR}/{idx}.jpg")]
    print(f"Total: {len(tasks)} images, {len(tasks) - len(pending)} cached, {len(pending)} to download")
    return pending


async def _fetch_one(session, idx, url, sem):
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
    """Download images with asyncio + aiohttp for maximum throughput."""
    sem = asyncio.Semaphore(ASYNC_BATCH)
    connector = aiohttp.TCPConnector(limit=ASYNC_BATCH, ttl_dns_cache=300, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        coros = [_fetch_one(session, idx, url, sem) for idx, url in tasks]
        results = []
        for coro in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Downloading"):
            results.append(await coro)
    ok = sum(results)
    print(f"Downloaded {ok}/{len(tasks)} images")


def download_sync(tasks):
    """Fallback: threaded downloads if aiohttp not available."""
    import requests

    def _dl(task):
        idx, url = task
        path = f"{IMAGES_DIR}/{idx}.jpg"
        try:
            resp = requests.get(url, timeout=ASYNC_TIMEOUT)
            if resp.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(resp.content)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as ex:
        list(tqdm(ex.map(_dl, tasks), total=len(tasks), desc="Downloading"))


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    download_csvs()
    tasks = collect_tasks()
    if not tasks:
        print("All images already cached.")
        return

    try:
        import aiohttp  # noqa: F811
        asyncio.run(download_async(tasks))
    except ImportError:
        print("aiohttp not installed, falling back to threaded downloads")
        download_sync(tasks)

    print("Done.")


if __name__ == "__main__":
    main()
