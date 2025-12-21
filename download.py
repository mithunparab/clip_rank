import os
import shutil
from huggingface_hub import hf_hub_download

def download_dataset():
    repo_id = "fast-stager/property-labels"
    filename = "dataset.csv" 
    
    print(f"Downloading {filename} from {repo_id}...")
    
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )
    
    destination = "dataset.csv"
    shutil.copy(local_path, destination)
    print(f"Dataset saved to: {destination}")

if __name__ == "__main__":
    download_dataset()