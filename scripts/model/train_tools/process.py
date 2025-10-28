
import pickle
import lzma
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
    
    for filePath in files_list:
        feature = []
        label = []
        with lzma.open(filePath, "rb") as file:
            datas = pickle.load(file)

            for data in datas:
                image_array = data.image
                croped_image_array = crop_image_ndarray(image_array, 0, 1, 0, 0.62)
                croped_resized_image_array = resize_image_ndarray(croped_image_array, target_size=(128, 128))
                croped_resized_image_array = normalize_image_ndarray(croped_resized_image_array)

                feature.append(croped_resized_image_array)
                label.append(data.current_controls)

    return feature, label

if __name__ == "__main__":
    process_datas()