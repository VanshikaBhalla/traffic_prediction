"""
Download METR-LA and PEMS-BAY datasets into data/ folder.
Sources: DCRNN (Li et al., ICLR 2018)
"""

import os
import requests
from tqdm import tqdm

# Dataset URLs from the official DCRNN data repo
DATA_URLS = {
    "metr-la.h5": "https://github.com/liyaguang/DCRNN_data/raw/master/metr-la.h5",
    "pems-bay.h5": "https://github.com/liyaguang/DCRNN_data/raw/master/pems-bay.h5",
}

def download_file(url, save_path):
    """Download with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(1024):
            size = f.write(data)
            bar.update(size)

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    for filename, url in DATA_URLS.items():
        save_path = os.path.join(data_dir, filename)
        if os.path.exists(save_path):
            print(f"✅ {filename} already exists, skipping...")
        else:
            print(f"⬇️ Downloading {filename} ...")
            download_file(url, save_path)
            print(f"✅ Saved to {save_path}")

if __name__ == "__main__":
    main()
