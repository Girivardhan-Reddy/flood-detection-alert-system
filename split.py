import os
import random
import shutil
from tqdm import tqdm
from PIL import Image

# -------- CONFIG --------
BASE_DIR = r"C:\Users\giriv\OneDrive\Desktop\working Projects\floods\Dataset"
OUTPUT_DIR = r"C:\Users\giriv\OneDrive\Desktop\working Projects\floods\Dataset_Split"

CLASSES = {
    "Flood Images": "Flood",
    "Non Flood Images": "NonFlood"
}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
IMG_SIZE = (224, 224)

# -------- CREATE FOLDERS --------
for split in ["train", "val", "test"]:
    for cls in CLASSES.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# -------- SPLIT + RESIZE --------
for src_folder, cls_name in CLASSES.items():
    src_path = os.path.join(BASE_DIR, src_folder)
    images = [f for f in os.listdir(src_path) if f.lower().endswith(IMG_EXTENSIONS)]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        print(f"Processing {cls_name} -> {split}")
        for file in tqdm(files):
            src_file = os.path.join(src_path, file)
            dst_file = os.path.join(OUTPUT_DIR, split, cls_name, file)

            try:
                with Image.open(src_file) as img:
                    img = img.convert("RGB")
                    img = img.resize(IMG_SIZE)
                    img.save(dst_file)
            except Exception as e:
                print(f"Error processing {file}: {e}")

print("Dataset split and resize completed successfully")
