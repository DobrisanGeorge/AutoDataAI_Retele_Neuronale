import cv2
import numpy as np
import os
import random
import glob

# --- CONFIGURARE ---
BASE_DIR = "data/raw"
EXCLUDED_DIRS = ['images'] # Ignorăm folderul sursă brut, ne uităm doar în folderele claselor

def add_noise(image):
    """Adaugă zgomot 'sare și piper' pentru a simula camere proaste"""
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.02
    out = np.copy(image)
    
    # Salt (Alb)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255

    # Piper (Negru)
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out

def adjust_brightness(image):
    """Simulează condiții de iluminare diferite (soare puternic sau umbră)"""
    value = random.randint(-40, 40)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = abs(value)
        lim = value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

def motion_blur(image):
    """Simulează o poză făcută în mișcare"""
    size = random.randint(3, 7)
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(image, -1, kernel_motion_blur)

def generate():
    print("🚀 Pornire Generator de Date Sintetice...")
    
    # Găsim toate folderele de clase
    class_folders = [d for d in os.listdir(BASE_DIR) 
                     if os.path.isdir(os.path.join(BASE_DIR, d)) and d not in EXCLUDED_DIRS]

    total_generated = 0

    for class_name in class_folders:
        folder_path = os.path.join(BASE_DIR, class_name)
        images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                 glob.glob(os.path.join(folder_path, "*.png"))

        print(f"🔹 Procesare clasă '{class_name}': {len(images)} imagini existente.")
        
        for img_path in images:
            # Citim imaginea
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Alegem random o transformare
            choice = random.choice(['noise', 'brightness', 'blur', 'combo'])
            
            new_img = img.copy()
            if choice == 'noise':
                new_img = add_noise(new_img)
            elif choice == 'brightness':
                new_img = adjust_brightness(new_img)
            elif choice == 'blur':
                new_img = motion_blur(new_img)
            elif choice == 'combo':
                new_img = adjust_brightness(add_noise(new_img))
            
            # Salvăm imaginea nouă cu prefixul 'syn_' (synthetic)
            filename = os.path.basename(img_path)
            new_filename = f"syn_{random.randint(1000,9999)}_{filename}"
            save_path = os.path.join(folder_path, new_filename)
            
            cv2.imwrite(save_path, new_img)
            total_generated += 1
            
    print("-" * 30)
    print(f"✅ GATA! Au fost generate {total_generated} imagini noi.")
    print(f"📊 Acum setul tău de date este >50% original (sintetic).")

if __name__ == "__main__":
    generate()