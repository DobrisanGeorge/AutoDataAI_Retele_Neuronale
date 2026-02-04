import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import time
import shutil

# --- 1. CONFIGURĂRI CĂI & DIRECTOARE ---
BASE_DIR = 'data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
DOCS_DIR = 'docs'
for d in [MODELS_DIR, RESULTS_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE = (260, 260) 

# --- 2. DEFINIRE EXPERIMENTE ---
EXPERIMENTS = [
    # NIVEL 1: Baseline Rapid (Fără augmentare, înghețat complet)
    {
        "name": "Exp1_SanityCheck",
        "lr": 1e-3, 
        "batch": 32, 
        "dropout": 0.2, 
        "epochs": 5,        
        "unfreeze": 0,      # Complet înghețat
        "augment": False
    },
    # NIVEL 2: Învățare Superficială (Dezghețăm doar vârful)
    {
        "name": "Exp2_Shallow_Tuning",
        "lr": 5e-4, 
        "batch": 32, 
        "dropout": 0.3, 
        "epochs": 10,
        "unfreeze": 10,
        "augment": False
    },
    # NIVEL 3: Învățare Medie (Introducem Augmentarea Ușoară)
    {
        "name": "Exp3_Mid_Range",
        "lr": 1e-4, 
        "batch": 16, 
        "dropout": 0.4, 
        "epochs": 15,
        "unfreeze": 30,     # Dezghețăm mediu
        "augment": True     # Pornim augmentarea
    },
    # NIVEL 4: High Performance (Ce aveam înainte ca 'Heavy')
    {
        "name": "Exp4_Deep_Dive",
        "lr": 1e-4, 
        "batch": 16, 
        "dropout": 0.5, 
        "epochs": 20,
        "unfreeze": 50,     # Dezghețăm adânc
        "augment": True
    },
    # NIVEL 5: Top Performance
    {
        "name": "Exp5_Final_Steroids",
        "lr": 5e-5,         # Learning rate foarte fin
        "batch": 16, 
        "dropout": 0.5, 
        "epochs": 25,       # Insistăm până iese perfect
        "unfreeze": 60,     # Dezghețăm masiv
        "augment": True
    }
]

# --- 3. FUNCȚII UTILITARE ---

def build_model(num_classes, config):
    """Construiește modelul dinamic în funcție de config."""
    print(f"   🏗️ Build Model: EfficientNetB0 | Unfreeze: {config['unfreeze']} layers | Dropout: {config['dropout']}")
    
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(260, 260, 3))
    
    base_model.trainable = True
    if config['unfreeze'] == 0:
        base_model.trainable = False
    else:
        # Înghețăm totul minus ultimele N straturi
        for layer in base_model.layers[:-config['unfreeze']]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(config['dropout'])(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(config['dropout'])(x)
    
    # Sigmoid pentru Multi-Label
    predictions = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def get_generators(config):
    """Returnează generatoarele de date (cu sau fără augmentare)."""
    if config['augment']:
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.1, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
        )
        
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=config['batch'], class_mode='categorical'
    )
    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=config['batch'], class_mode='categorical'
    )
    
    return train_gen, val_gen

def plot_final_results(history, y_true, y_pred, classes):
    """Generează toate graficele necesare pentru documentație."""
    # 1. Matrice Confuzie
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Final Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(DOCS_DIR, 'confusion_matrix_optimized.png'))
    plt.close()

    # 2. Learning Curves
    if history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Val')
        plt.title('Acuratețe')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()
        plt.savefig(os.path.join(DOCS_DIR, 'learning_curves_final.png'))
        plt.close()

def main():
    print("🚀 START: PROTOCOL 'TRAIN ON STEROIDS' ...")
    
    # 1. Identificare Clase
    temp_gen = ImageDataGenerator().flow_from_directory(TRAIN_DIR, target_size=(10,10), batch_size=1)
    class_names = list(temp_gen.class_indices.keys())
    num_classes = len(class_names)
    
    # Salvare classes.txt (CRITIC)
    with open(os.path.join(MODELS_DIR, 'classes.txt'), 'w') as f:
        f.write('\n'.join(class_names))
    print(f"✅ Clase detectate ({num_classes}): {class_names}")

    results_log = []
    best_acc = 0.0
    best_exp_name = ""
    best_history = None

    # 2. LOOP EXPERIMENTE
    for exp in EXPERIMENTS:
        print(f"\n🧪 --- Rulare Experiment: {exp['name']} ---")
        start_time = time.time()
        
        # Pregătire
        train_gen, val_gen = get_generators(exp)
        model = build_model(num_classes, exp)
        
        model.compile(optimizer=Adam(learning_rate=exp['lr']), 
                      loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks (Early Stopping ca să nu pierdem timp degeaba)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
        ]
        
        # Calculare Class Weights (Pentru dataset dezechilibrat)
        try:
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced', classes=np.unique(train_gen.classes), y=train_gen.classes
            )
            cw_dict = dict(enumerate(class_weights))
        except:
            cw_dict = None

        # Antrenare
        history = model.fit(
            train_gen, 
            validation_data=val_gen, 
            epochs=exp['epochs'], 
            callbacks=callbacks,
            class_weight=cw_dict,
            verbose=1
        )
        
        # Evaluare rapidă
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        duration = (time.time() - start_time) / 60
        
        print(f"🏁 Rezultat {exp['name']}: Val Acc = {val_acc:.2%} | Timp: {duration:.1f} min")
        
        # Logare
        results_log.append({
            "Experiment": exp['name'],
            "Learning Rate": exp['lr'],
            "Batch": exp['batch'],
            "Dropout": exp['dropout'],
            "Augment": exp['augment'],
            "Val Accuracy": round(val_acc, 4),
            "Val Loss": round(val_loss, 4),
            "Duration (min)": round(duration, 2)
        })
        
        # Check if BEST
        if val_acc > best_acc:
            print(f"⭐ NEW BEST MODEL! ({val_acc:.2%}) -> Salvăm temporar...")
            best_acc = val_acc
            best_exp_name = exp['name']
            best_history = history
            # Salvăm modelul direct ca 'optimized_model.h5'
            model.save(os.path.join(MODELS_DIR, 'optimized_model.h5'))

    # 3. CONCLUZII & ARTEFACTE FINALE
    print(f"\n🏆 CÂȘTIGĂTOR: {best_exp_name} cu {best_acc:.2%}")
    
    # Salvare CSV Experimente
    pd.DataFrame(results_log).to_csv(os.path.join(RESULTS_DIR, 'optimization_experiments.csv'), index=False)
    
    # Evaluare Finală pe Setul de Test
    final_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'optimized_model.h5'))
    
    # Generator Test
    test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=16, class_mode='categorical', shuffle=False
    )
    
    # Predicții Finale
    test_loss, test_acc = final_model.evaluate(test_gen)
    preds = final_model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    
    # Salvare Metrici JSON
    metrics = {
        "model_name": "optimized_model.h5",
        "best_experiment": best_exp_name,
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4)
    }
    with open(os.path.join(RESULTS_DIR, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Salvare Istoric Training & Grafice Finale
    if best_history:
        pd.DataFrame(best_history.history).to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)
        plot_final_results(best_history, y_true, y_pred, class_names)
    
    # Analiză Erori
    print("⚠️ Generare Analiză Erori...")
    errors = []
    filenames = test_gen.filenames
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            errors.append({
                "File": filenames[i],
                "True": class_names[y_true[i]],
                "Pred": class_names[y_pred[i]],
                "Conf": round(float(np.max(preds[i])), 4)
            })
    pd.DataFrame(errors).to_csv(os.path.join(RESULTS_DIR, 'error_analysis.csv'), index=False)
    
    print("\n✅ TOATE SISTEMELE PREGĂTITE. Poți rula App.py!")

if __name__ == "__main__":
    main()