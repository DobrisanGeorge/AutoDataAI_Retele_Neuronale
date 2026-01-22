import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def preprocess_image_for_model(image, target_size=(224, 224)):
    """
    Pregătește o imagine PIL pentru a intra în model (MobileNetV2/EfficientNet).
    1. Resize cu păstrarea aspectului (Fit)
    2. Transformare în array
    3. Normalizare sau Preprocesare specifică
    4. Adăugare dimensiune batch
    """
    # 1. Resize inteligent
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    
    # 2. Conversie la array
    img_array = np.asarray(image)
    
    # 3. Preprocesare (Aici alegi în funcție de model)
    # Varianta 1: Pentru MobileNetV2 standard (0-1)
    # img_array = (img_array.astype(np.float32) / 255.0)
    
    # Varianta 2: Pentru EfficientNet (folosind funcția din Keras)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # 4. Expand dims (H, W, C) -> (1, H, W, C)
    img_reshape = img_array[np.newaxis, ...]
    
    return img_reshape