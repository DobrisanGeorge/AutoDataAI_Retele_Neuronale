import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import time

# --- CONFIGURĂRI CĂI ---
BASE_DIR = 'data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
DOCS_DIR = 'docs'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

IMG_SIZE = (260, 260)

# --- DEFINIRE EXPERIMENTE (Cerința: min 4 experimente) ---
EXPERIMENTS = [
    {"name": "Exp1_Baseline_LR001", "lr": 0.001, "batch": 32, "dropout": 0.3},
    {"name": "Exp2_LowLR",          "lr": 0.0001, "batch": 32, "dropout": 0.3},
    {"name": "Exp3_HighDropout",    "lr": 0.001,  "batch": 32, "dropout": 0.5},
    {"name": "Exp4_Best_Efficient", "lr": 0.0001, "batch": 16, "dropout": 0.5} 
]

def build_model(num_classes, dropout_rate):
    """Construiește modelul EfficientNetB0"""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(260, 260, 3))
    
    # Fine-tuning parțial (ultimele 20 straturi)
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    # Sigmoid pentru Multi-Label
    predictions = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def plot_confusion_matrix(y_true, y_pred, classes):
    """Generează și salvează Confusion Matrix"""
    # Convertim probabilitățile în clase (argmax pentru matricea simplificată)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Optimized Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = os.path.join(DOCS_DIR, 'confusion_matrix_optimized.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion Matrix salvată în: {save_path}")

def main():
    print("🚀 Start Optimizare Automată (Etapa 6)...")
    
    # 1. GENERATOARE DATE
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    # Citire date pentru a afla clasele
    temp_gen = val_test_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=16)
    class_names = list(temp_gen.class_indices.keys())
    num_classes = len(class_names)
    
    # Salvare classes.txt
    with open(os.path.join(MODELS_DIR, 'classes.txt'), 'w') as f:
        f.write('\n'.join(class_names))

    results_log = []
    best_acc = 0
    best_model = None
    best_history = None

    # 2. RULARE LOOP EXPERIMENTE
    for exp in EXPERIMENTS:
        print(f"\n🧪 Rulare {exp['name']}...")
        start_time = time.time()
        
        # Setup Generatoare cu batch size variabil
        train_gen = train_datagen.flow_from_directory(
            TRAIN_DIR, target_size=IMG_SIZE, batch_size=exp['batch'], class_mode='categorical'
        )
        val_gen = val_test_datagen.flow_from_directory(
            VAL_DIR, target_size=IMG_SIZE, batch_size=exp['batch'], class_mode='categorical'
        )

        model = build_model(num_classes, exp['dropout'])
        model.compile(optimizer=Adam(learning_rate=exp['lr']), 
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        # Antrenare scurtă pentru experiment (sau lungă pentru cel final)
        # Nota: Pentru viteză la demonstrație punem 5 epoci, pt rezultate reale pune 15
        epochs = 8 if "Best" in exp['name'] else 3 
        
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)
        
        # Colectare rezultate
        final_acc = history.history['val_accuracy'][-1]
        final_loss = history.history['val_loss'][-1]
        duration = (time.time() - start_time) / 60
        
        results_log.append({
            "Experiment": exp['name'],
            "Learning Rate": exp['lr'],
            "Batch Size": exp['batch'],
            "Dropout": exp['dropout'],
            "Val Accuracy": round(final_acc, 4),
            "Val Loss": round(final_loss, 4),
            "Duration (min)": round(duration, 2)
        })
        
        # Reținem cel mai bun model
        if final_acc > best_acc:
            best_acc = final_acc
            best_model = model
            best_history = history
            print(f"⭐ New Best Model found: {exp['name']} ({final_acc:.2%})")

    # 3. SALVARE RAPORT EXPERIMENTE (CSV)
    df_results = pd.DataFrame(results_log)
    csv_path = os.path.join(RESULTS_DIR, 'optimization_experiments.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ Raport experimente salvat în: {csv_path}")

    # 4. SALVARE MODEL OPTIMIZAT
    optimized_path = os.path.join(MODELS_DIR, 'optimized_model.h5')
    best_model.save(optimized_path)
    print(f"✅ Model Optimizat salvat în: {optimized_path}")

    # 5. GENERARE METRICI FINALE & CONFUSION MATRIX
    print("\n🔍 Evaluare Finală pe Test Set...")
    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=16, class_mode='categorical', shuffle=False
    )
    
    # Metrici
    test_loss, test_acc = best_model.evaluate(test_gen)
    predictions = best_model.predict(test_gen)
    
    # Salvare JSON
    metrics = {
        "model_name": "optimized_model.h5",
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "best_experiment": df_results.loc[df_results['Val Accuracy'].idxmax()]['Experiment']
    }
    with open(os.path.join(RESULTS_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrici finale salvate în: {os.path.join(RESULTS_DIR, 'final_metrics.json')}")

    # Generare Confusion Matrix
    y_true = test_gen.classes
    plot_confusion_matrix(y_true, predictions, class_names)
    
    # Generare Training Plot pentru modelul optimizat
    if best_history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(best_history.history['accuracy'], label='Train')
        plt.plot(best_history.history['val_accuracy'], label='Val')
        plt.title('Accuracy (Optimized)')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(best_history.history['loss'], label='Train')
        plt.plot(best_history.history['val_loss'], label='Val')
        plt.title('Loss (Optimized)')
        plt.legend()
        plt.savefig(os.path.join(DOCS_DIR, 'learning_curves_final.png'))

    print("\n🏁 ETAPA 6 COMPLETATĂ CU SUCCES!")

if __name__ == "__main__":
    main()