# rallyrobopilot/ml/training.py
import torch
import numpy as np
import lzma
import pickle
import glob
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .dataset import RallyDataset # Note the relative import

    

class SensingSnapshot:
    pass
def prepare_data(batch_size=32, test_size=0.2, sources=['normal'], include_speed=False, remove_outliers=False):
    """
    Loads .npz recordings, processes them, and returns DataLoaders and normalization statistics.

    Args:
        batch_size (int): The batch size for the DataLoader.
        test_size (float): The proportion of the dataset for validation.
        sources (list): Subdirectories in 'data' to load from.
        include_speed (bool): If True, adds car speed as a 16th feature.
        remove_outliers (bool): If True, removes samples with a Z-score > 3.0.
    """
    print(f"--- Starting Data Preparation for sources: {sources} ---")

    # --- Configuration ---
    AUGMENTATION_FACTOR = 3
    NOISE_LEVEL = 5
    MIRROR_FOLDERS = {"mirrored"} # Folders to apply mirroring to
    # --- End Configuration ---

    def all_controls_zero(snap):
        return all(v == 0 for v in snap.current_controls)

    def trim_edges(snaps_list):
        if not snaps_list:
            return [], 0, 0
        start = 0
        while start < len(snaps_list) and all_controls_zero(snaps_list[start]):
            start += 1
        end = len(snaps_list) - 1
        while end >= 0 and all_controls_zero(snaps_list[end]):
            end -= 1
        if start > end:
            return [], start, len(snaps_list) - end - 1
        return snaps_list[start:end+1], start, len(snaps_list) - end - 1

    def mirror_in_place(snap):
        cc = list(snap.current_controls)
        if len(cc) >= 4:
            cc[2], cc[3] = cc[3], cc[2]
        snap.current_controls = tuple(cc)
        snap.raycast_distances = snap.raycast_distances[::-1].copy() \
            if hasattr(snap.raycast_distances, 'copy') else snap.raycast_distances[::-1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_base_path = os.path.join(project_root, "data")

    all_snapshots = []
    for source_folder in sources:
        search_path = os.path.join(data_base_path, source_folder, "*.npz")
        file_paths = glob.glob(search_path)
        if not file_paths:
            print(f"[!] Warning: No data found in '{data_base_path}/{source_folder}'")
            continue

        do_mirror = (source_folder in MIRROR_FOLDERS)
        for file_path in file_paths:
            print(f"Loading data from: {file_path}")
            with lzma.open(file_path, "rb") as f:
                record = pickle.load(f)

            trimmed, removed_head, removed_tail = trim_edges(record)
            if not trimmed:
                print(f"  -> Record is idle-only. Skipped.")
                continue

            if do_mirror:
                for snap in trimmed:
                    mirror_in_place(snap)

            all_snapshots.extend(trimmed)
            print(f"  -> Kept {len(trimmed)} snapshots (removed head={removed_head}, tail={removed_tail}, mirror={do_mirror})")

    if not all_snapshots:
        print("\n[!] No data was loaded! Check the 'sources' argument.")
        return None, None, None, None

    print(f"\nTotal snapshots after trimming: {len(all_snapshots)}")

    if include_speed:
        print("Assembling features with raycasts and car speed (16 features).")
        features = [list(s.raycast_distances) + [s.car_speed] for s in all_snapshots]
    else:
        print("Assembling features with raycasts only (15 features).")
        features = [s.raycast_distances for s in all_snapshots]

    labels = [s.current_controls for s in all_snapshots]

    X = np.array(features, dtype=np.float32)
    Y = np.array(labels, dtype=np.float32)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=42, shuffle=True
    )

    if AUGMENTATION_FACTOR > 1:
        print(f"Augmenting training data by a factor of {AUGMENTATION_FACTOR}...")
        X_train_augmented = [X_train]
        Y_train_augmented = [Y_train]
        for _ in range(AUGMENTATION_FACTOR - 1):
            noise = np.random.normal(loc=0.0, scale=NOISE_LEVEL, size=X_train.shape)
            noisy_X = np.clip(X_train + noise, 0, 100000).astype(np.float32)
            X_train_augmented.append(noisy_X)
            Y_train_augmented.append(Y_train)
        X_train = np.concatenate(X_train_augmented, axis=0)
        Y_train = np.concatenate(Y_train_augmented, axis=0)
        print(f"Augmentation complete. New training set size: {len(X_train)}")

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    print("\nNormalization stats (Mean/Std) calculated from training data.")
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    if remove_outliers:
        print("\n--- Removing Outlier Samples ---")
        outlier_samples_mask_train = np.any(np.abs(X_train) > 3.0, axis=1)
        num_outlier_samples_train = np.sum(outlier_samples_mask_train)
        if num_outlier_samples_train > 0:
            X_train = X_train[~outlier_samples_mask_train]
            Y_train = Y_train[~outlier_samples_mask_train]
            print(f"✅ Removed {num_outlier_samples_train} samples from the training set.")

        outlier_samples_mask_val = np.any(np.abs(X_val) > 3.0, axis=1)
        num_outlier_samples_val = np.sum(outlier_samples_mask_val)
        if num_outlier_samples_val > 0:
            X_val = X_val[~outlier_samples_mask_val]
            Y_val = Y_val[~outlier_samples_mask_val]
            print(f"✅ Removed {num_outlier_samples_val} samples from the validation set.")
        print("--------------------------------")

    print(f"\nData split into {len(X_train)} training samples and {len(X_val)} validation samples.")

    train_dataset = RallyDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = RallyDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

    num_cpu_cores = os.cpu_count()
    workers_to_use = num_cpu_cores // 2 if num_cpu_cores else 4 # Utilise la moitié des coeurs logiques

    print(f"Using {workers_to_use} workers for data loading.")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print("--- Data Preparation Complete ---")
    return train_loader, val_loader, mean, std
    