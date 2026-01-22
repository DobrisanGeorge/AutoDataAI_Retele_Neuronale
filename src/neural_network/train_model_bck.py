import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

# --- CONFIGURĂRI ---
BASE_DIR = 'data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Directoare pentru salvare rezultate
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parametri
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001

def plot_history(history):
    """Generează și salvează graficele de Loss și Accuracy"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Grafic Acuratețe
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Grafic Eroare (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    save_path = os.path.join(RESULTS_DIR, 'training_plot.png')
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Grafic salvat în: {save_path}")

def train():
    # 1. GENERATOARE DE DATE
    print("🔄 Pregătire date...")
    
    # Augmentare doar pe Train
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Doar rescaling pentru Val și Test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    validation_generator = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )

    # 2. SALVAREA CLASELOR (Foarte important pentru UI)
    class_names = list(train_generator.class_indices.keys())
    classes_path = os.path.join(MODELS_DIR, 'classes.txt')
    with open(classes_path, 'w') as f:
        f.write('\n'.join(class_names))
    print(f"✅ Lista claselor salvată în: {classes_path}")

    # 3. DEFINIRE MODEL (MobileNetV2)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Înghețăm baza

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # 4. ANTRENARE
    print("🚀 Începe antrenarea...")
    # Oprim antrenarea dacă nu se îmbunătățește timp de 3 epoci (Early Stopping)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # 5. SALVARE REZULTATE
    # Salvare Model
    model_path = os.path.join(MODELS_DIR, 'damage_model.h5')
    model.save(model_path)
    print(f"💾 Model salvat în: {model_path}")

    # Salvare Istoric (CSV)
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(RESULTS_DIR, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"📄 Istoric salvat în: {history_path}")

    # Generare Grafice
    plot_history(history)

    # 6. EVALUARE FINALĂ PE TEST SET
    print("🔍 Evaluare pe setul de test...")
    test_loss, test_acc = model.evaluate(test_generator)
    
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4)
    }
    metrics_path = os.path.join(RESULTS_DIR, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✅ Rezultate finale: Acuratețe={test_acc*100:.2f}%, Loss={test_loss:.4f}")
    print("🏁 Proces complet!")

if __name__ == "__main__":
    train()