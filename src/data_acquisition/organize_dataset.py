import pandas as pd
import os
import shutil

# --- CONFIGURARE ---
BASE_DIR = "data/raw"
CSV_PATH = os.path.join(BASE_DIR, "data.csv")
SOURCE_IMAGES_DIR = os.path.join(BASE_DIR, "images")

def organize():
    # 1. Verificări fișiere
    if not os.path.exists(CSV_PATH):
        print(f"❌ EROARE: Nu găsesc {CSV_PATH}")
        return
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"❌ EROARE: Nu găsesc folderul {SOURCE_IMAGES_DIR}")
        print(f"   Asigură-te că ai dezarhivat pozele în: {os.path.abspath(SOURCE_IMAGES_DIR)}")
        return

    print("⏳ Citesc CSV-ul...")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Eroare citire CSV: {e}")
        return

    # 2. Identificare Coloane
    # În CSV-ul tău coloanele par a fi 'image' și 'classes'
    cols = df.columns.tolist()
    img_col = next((c for c in cols if 'image' in c.lower()), None)
    lbl_col = next((c for c in cols if 'class' in c.lower() or 'label' in c.lower()), None)

    if not img_col or not lbl_col:
        print(f"❌ Nu am identificat coloanele. Găsit: {cols}")
        return

    print(f"✅ Folosesc: Imagine='{img_col}', Etichetă='{lbl_col}'")

    moved = 0
    missing = 0

    # 3. Mutarea Imaginilor
    for _, row in df.iterrows():
        # Din CSV vine 'image/0.jpeg', noi vrem doar '0.jpeg'
        raw_filename = str(row[img_col])
        filename = os.path.basename(raw_filename) 
        
        label = str(row[lbl_col]).strip().replace(" ", "_")

        # Calea sursă (unde sunt pozele acum)
        src = os.path.join(SOURCE_IMAGES_DIR, filename)
        
        # Calea destinație (folderul clasei)
        dst_folder = os.path.join(BASE_DIR, label)
        dst = os.path.join(dst_folder, filename)

        if os.path.exists(src):
            os.makedirs(dst_folder, exist_ok=True)
            shutil.copy(src, dst)
            moved += 1
        else:
            # Încercăm și variante de extensie, just in case
            if filename.endswith('.jpeg'):
                alt_name = filename.replace('.jpeg', '.jpg')
                src_alt = os.path.join(SOURCE_IMAGES_DIR, alt_name)
                if os.path.exists(src_alt):
                    os.makedirs(dst_folder, exist_ok=True)
                    shutil.copy(src_alt, os.path.join(dst_folder, alt_name))
                    moved += 1
                else:
                    missing += 1
            else:
                missing += 1

    print(f"✅ GATA! Mutate: {moved}. Lipsă: {missing}.")
    print(f"   (E normal să fie câteva lipsă dacă CSV-ul are intrări vechi, dar majoritatea trebuie mutate)")

if __name__ == "__main__":
    organize()