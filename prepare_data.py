import pandas as pd
import requests
import os
from io import BytesIO
from PIL import Image
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

def main():
    if not os.path.exists('images'):
        os.makedirs('images')
        
    df = pd.read_csv('dataset.csv')
    
    tasks = list(zip(df.index, df['url']))
    
    print(f"Downloading {len(tasks)} images...")
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(download_one, tasks), total=len(tasks)))
        
    print("Done. Images cached in /images folder.")

if __name__ == "__main__":
    main()