import pickle
import lzma
import os
from PIL import Image
import tqdm

def all_controls_zero(snap):
    return all(v == 0 for v in snap.current_controls)

def data_cleaner(path):
    with lzma.open(path, "rb") as file:
        data = pickle.load(file)
    
    start = 0
    while start < len(data) and all_controls_zero(data[start]):
        start += 1
    
    end = len(data) - 1
    while end >= 0 and all_controls_zero(data[end]):
        end -= 1

    if start == 0 and end == len(data) - 1:
        print(f"Already clean → skipped: {path}")
        return

    data = data[start:end+1] if start <= end else []

    with lzma.open(path, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Cleaned: {path}")

def clean_all_npz_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            full_path = os.path.join(directory, filename)
            data_cleaner(full_path)

    print("All files cleaned.")

# Lance le nettoyage dans le dossier courant :
clean_all_npz_in_directory(".")
