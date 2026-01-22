import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Configurare stil
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Căi
CSV_PATH = 'results/training_history.csv'
DOCS_DIR = 'docs'
os.makedirs(DOCS_DIR, exist_ok=True)

def generate_plots():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Eroare: Nu găsesc {CSV_PATH}. Rulează mai întâi antrenarea!")
        return

    # Citire date
    df = pd.read_csv(CSV_PATH)
    epochs = range(1, len(df) + 1)

    # 1. GENERARE LOSS CURVE (Curba de Eroare)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, df['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    plt.title('Loss Curve - EfficientNet Optimization', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Binary Crossentropy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path_loss = os.path.join(DOCS_DIR, 'learning_curves_final.png')
    plt.savefig(save_path_loss, dpi=300)
    print(f"✅ Loss Curve salvat în: {save_path_loss}")
    plt.close()

    # 2. GENERARE ACCURACY CURVE
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['accuracy'], 'g-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, df['val_accuracy'], 'orange', linestyle='--', label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy Evolution', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    save_path_acc = os.path.join(DOCS_DIR, 'accuracy_curve.png')
    plt.savefig(save_path_acc, dpi=300)
    print(f"✅ Accuracy Curve salvat în: {save_path_acc}")
    plt.close()

if __name__ == "__main__":
    generate_plots()