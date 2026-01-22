import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

# --- CONFIGURĂRI ---
BASE_DIR = 'data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dimensiuni pentru EfficientNetB0
IMG_SIZE = (260, 260)
BATCH_SIZE = 16 

def plot_history(history):
    """Generează graficul training_plot_pro.png"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    # Grafic Acuratețe
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Model Accuracy (EfficientNet)')
    plt.legend()

    # Grafic Eroare
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    
    save_path = os.path.join(RESULTS_DIR, 'training_plot_pro.png')
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Grafic salvat în: {save_path}")

def train():
    print("🚀 Inițializare Antrenare EfficientNet (Multi-Label)...")
    
    # 1. GENERATOARE DATE
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    print("📥 Încărcare imagini...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    validation_generator = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )
    
    # Salvare clase
    class_names = list(train_generator.class_indices.keys())
    with open(os.path.join(MODELS_DIR, 'classes.txt'), 'w') as f:
        f.write('\n'.join(class_names))

    # Calcul Ponderi (Opțional, pentru echilibrare)
    try:
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(train_generator.classes), 
            y=train_generator.classes
        )
        class_weights_dict = dict(enumerate(class_weights))
    except:
        class_weights_dict = None
        print("⚠️ Nu s-au putut calcula ponderile. Se continuă fără.")

    # 2. DEFINIRE MODEL
    print("\n🧠 Construire Model...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(260, 260, 3))
    
    # Fine-Tuning: Înghețăm tot, lăsăm doar ultimele 20 layere libere
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # IMPORTANT: 'sigmoid' pentru Multi-Label (mai multe daune simultan)
    predictions = Dense(len(class_names), activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # 3. ANTRENARE
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'damage_model.h5'), save_best_only=True, verbose=1)
    ]

    print("🔥 Start Epoci...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=15, # Ajustează dacă durează prea mult
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    # 4. SALVARE REZULTATE (Aici se creează fișierele lipsă!)
    # Istoric CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)
    print("✅ Istoric salvat în training_history.csv")
    
    # Grafic PNG
    plot_history(history)
    
    # Evaluare Test JSON
    print("\n🔍 Evaluare finală...")
    results = model.evaluate(test_generator)
    metrics = {"test_loss": results[0], "test_accuracy": results[1]}
    
    with open(os.path.join(RESULTS_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("✅ Metrici salvate în test_metrics.json")

    print("\n🏁 GATA! Acum poți rula 'streamlit run app.py'")

if __name__ == "__main__":
    train()