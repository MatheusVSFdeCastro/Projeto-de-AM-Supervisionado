import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

CLASSES_NOMES = [
    "Carcinoma Basocelular (Maligno)", "Dermatofibroma (Benigno)", 
    "Lesao Vascular (Benigno)", "Melanoma (Maligno)", 
    "Nevo Melanocitico (Benigno)", "Queratose Actinica (Pre-Maligno)", 
    "Queratose Benigna (Benigno)"
]

def construir_arquitetura():
    """Recria a planta estrutural do modelo"""
    inputs = Input(shape=(224, 224, 3))
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(7, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def carregar_modelo_completo(caminho_pesos="pesos_skin_cancer.weights.h5"):
    """Instancia o esqueleto e injeta os números (pesos)"""
    model = construir_arquitetura()
    model.load_weights(caminho_pesos)
    return model

def predizer_imagem(caminho_img, model):
    """Sua lógica otimizada com o limiar médico de 0.35"""
    img_raw = tf.io.read_file(caminho_img)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [224, 224]) / 255.0
    batch_img = tf.expand_dims(img, axis=0)
    
    probabilidades = model.predict(batch_img, verbose=0)[0]
    
    idx_carcinoma, idx_melanoma = 0, 3
    if probabilidades[idx_melanoma] > 0.35:
        classe_idx = idx_melanoma
    elif probabilidades[idx_carcinoma] > 0.35:
        classe_idx = idx_carcinoma
    else:
        classe_idx = np.argmax(probabilidades)
        
    return CLASSES_NOMES[classe_idx], probabilidades[classe_idx]
