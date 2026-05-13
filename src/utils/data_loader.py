"""
Módulo para cargar y procesar datos de imágenes
"""

import os
import numpy as np
import cv2
from skimage.io import imread
from PIL import Image
from sklearn.utils import shuffle
from .constants import TRAIN_PATH, TEST_PATH, CATEGORIES, IMAGE_WIDTH, IMAGE_HEIGHT


def read_data(ruta, grayscale=False):
    """
    Carga imágenes de una carpeta y las redimensiona.
    
    Args:
        ruta (str): Ruta a la carpeta con las imágenes
        grayscale (bool): Si True, convierte a escala de grises
    
    Returns:
        tuple: (X, y) arrays de numpy con imágenes y etiquetas
    """
    X = []
    Y = []
    
    for c, cat in enumerate(CATEGORIES):
        path = os.path.join(ruta, cat)
        
        if not os.path.exists(path):
            print(f"Advertencia: ruta no encontrada {path}")
            continue
        
        for file in os.listdir(path):
            if not file.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue
            
            try:
                if grayscale:
                    # Cargar en escala de grises
                    img = Image.open(os.path.join(path, file))
                    
                    if img.mode in ('P', 'PA'):
                        img = img.convert('RGBA')
                    
                    img = img.convert('L')
                    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                    
                    arr = np.array(img).astype(np.float32) / 255.0
                    # Repetir el canal para compatibilidad con modelos RGB
                    arr = np.stack([arr] * 3, axis=-1)
                else:
                    # Cargar en color
                    image = imread(os.path.join(path, file))
                    img_small = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    
                    # Verificar que tenga 3 canales
                    if img_small.shape != (IMAGE_WIDTH, IMAGE_HEIGHT, 3):
                        print(f"Shape inesperado {img_small.shape} → {os.path.join(path, file)}")
                        continue
                    
                    arr = img_small.astype(np.float32) / 255.0
                
                X.append(arr)
                Y.append(c)
                
            except Exception as e:
                print(f"Error procesando {file}: {e}")
                continue
    
    return np.array(X), np.array(Y)


def load_and_prepare_data(grayscale=False, random_state=42):
    """
    Carga datos de train y test, normaliza y baraja.
    
    Args:
        grayscale (bool): Si True, carga imágenes en escala de grises
        random_state (int): Seed para reproducibilidad
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("Cargando datos de train...")
    X_train, y_train = read_data(TRAIN_PATH, grayscale=grayscale)
    
    print("Cargando datos de test...")
    X_test, y_test = read_data(TEST_PATH, grayscale=grayscale)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Mezclar datos de entrenamiento
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def save_data_npz(X_train, X_test, y_train, y_test, filepath):
    """
    Guarda los datos en un archivo .npz para carga rápida posterior.
    
    Args:
        X_train, X_test, y_train, y_test: Arrays de datos
        filepath (str): Ruta donde guardar el archivo
    """
    np.savez(filepath,
             X_train=X_train,
             y_train=y_train,
             X_test=X_test,
             y_test=y_test)
    print(f"Datos guardados en {filepath}")


def load_data_npz(filepath):
    """
    Carga datos desde un archivo .npz.
    
    Args:
        filepath (str): Ruta del archivo .npz
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    data = np.load(filepath)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


def preprocess_image(image_path, size=(IMAGE_WIDTH, IMAGE_HEIGHT), grayscale=False):
    """
    Preprocesa una imagen individual para predicción.
    
    Args:
        image_path (str): Ruta de la imagen
        size (tuple): Tamaño de salida
        grayscale (bool): Si True, convierte a escala de grises
    
    Returns:
        tuple: (imagen PIL, array para predicción)
    """
    img = Image.open(image_path).convert('RGB')
    
    if grayscale:
        img = img.convert('L')
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.stack([arr] * 3, axis=-1)
    else:
        arr = np.array(img.resize(size)).astype(np.float32) / 255.0
    
    arr = np.expand_dims(arr, axis=0)
    return img, arr
