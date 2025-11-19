import os
import numpy as np
import cv2
import glob

def crop_image_ndarray(image_array, left_crop_ratio, right_crop_ratio, top_crop_ratio, bot_crop_ratio):
    height, width, channels = image_array.shape

    left_crop = int(width * left_crop_ratio)
    right_crop = int(width * right_crop_ratio)
    top_crop = int(height * top_crop_ratio)
    bot_crop = int(height * bot_crop_ratio)

    cropped_image = image_array[top_crop:bot_crop, left_crop:right_crop, :]

    return cropped_image

def resize_image_ndarray(image_array, target_size):
    resized = cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)
    return resized

def normalize_image_ndarray(image_array):
    normalized = image_array.astype('float32') / 255.0
    return normalized

def process_datas(files_path="record_*.npz"):
    files_list = sorted(glob.glob(files_path))
    print(f"Found {len(files_list)} files to process.")

    features = []
    labels = []

    forward_count = 0
    left_count = 0
    right_count = 0

    for filePath in files_list:
        ext = os.path.splitext(filePath)[1].lower()
        if ext != ".npz":
            continue

        d = np.load(filePath, allow_pickle=False)
        if "images" not in d or "controls" not in d:
            print(f"[skip] {filePath} missing 'images' or 'controls'")
            continue

        imgs = d["images"]    # (N, H, W, 3) uint8
        ctrls = d["controls"] # (N, 4)

        if len(imgs) == 0:
            continue

        imgs_normalized = normalize_image_ndarray(imgs)

        features.append(imgs_normalized)
        labels.append(ctrls)

        forward_count = np.sum(ctrls[:,0])
        left_count    = np.sum(ctrls[:,2])
        right_count   = np.sum(ctrls[:,3])

    print(f"Total forward commands: {forward_count}")
    print(f"Total left commands: {left_count}")
    print(f"Total right commands: {right_count}")
    print(f"Loaded {len(features)} files.")
    return features, labels


if __name__ == "__main__":
    process_datas()