import os
import shutil
import random

# --- CONFIGURARE ---
RAW_DIR = "data/raw"
BASE_DIR = "data"
# Ignorăm folderul 'images' și fișierul csv, vrem doar folderele claselor
CLASSES = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d)) and d != 'images']

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split():
    if not CLASSES:
        print("❌ Nu am găsit foldere cu clase în data/raw. Rulează organize_dataset.py întâi!")
        return

    print(f"ℹ️ Clase găsite: {CLASSES}")

    for class_name in CLASSES:
        os.makedirs(os.path.join(BASE_DIR, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, 'validation', class_name), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, 'test', class_name), exist_ok=True)

        src_path = os.path.join(RAW_DIR, class_name)
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        print(f" -> {class_name}: {len(train_imgs)} Train, {len(val_imgs)} Val, {len(test_imgs)} Test")

        for img in train_imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(BASE_DIR, 'train', class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(BASE_DIR, 'validation', class_name, img))
        for img in test_imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(BASE_DIR, 'test', class_name, img))

    print("✅ Split complet!")

if __name__ == "__main__":
    split()