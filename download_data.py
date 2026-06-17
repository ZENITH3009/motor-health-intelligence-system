import os
import urllib.request
import ssl

# Ignore SSL certificate mismatch for this academic server
ssl._create_default_https_context = ssl._create_unverified_context

# UPDATED: The new official CWRU repository URL base (engineering.case.edu)
BASE_URL = "https://engineering.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/"

# The exact files you are missing based on your data_loader.py
FILES = {
    'inner_race': ['105.mat', '106.mat', '107.mat', '108.mat', '169.mat', '170.mat', '171.mat', '172.mat', '209.mat', '210.mat', '211.mat', '212.mat'],
    'outer_race': ['130.mat', '131.mat', '132.mat', '197.mat', '198.mat', '199.mat'],
    'ball': ['118.mat', '119.mat', '120.mat', '121.mat', '185.mat', '186.mat', '187.mat', '188.mat']
}

print("Starting CWRU dataset download from updated server...")

for folder, files in FILES.items():
    folder_path = f"data/raw/{folder}"
    os.makedirs(folder_path, exist_ok=True)
    
    for file in files:
        file_path = f"{folder_path}/{file}"
        
        if not os.path.exists(file_path):
            url = BASE_URL + file
            print(f"Downloading {file} into {folder}/...")
            try:
                urllib.request.urlretrieve(url, file_path)
            except Exception as e:
                print(f"  -> Failed to download {file}: {e}")
        else:
            print(f"File {file} already exists in {folder}/, skipping.")

print("\nDownload script finished! Check your folders.")