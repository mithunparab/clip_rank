import os
import shutil
from huggingface_hub import hf_hub_download

def download_dataset():
    repo_id = "fast-stager/property-labels"
    filename = "annotations.csv" 
    
    print(f"Downloading {filename} from {repo_id}...")
    
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )
        
        destination = "dataset.csv"
        shutil.copy(local_path, destination)
        print(f"Success! Saved locally as: {destination}")
        
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Make sure you are logged in if the repo is private: 'huggingface-cli login'")

if __name__ == "__main__":
    download_dataset()