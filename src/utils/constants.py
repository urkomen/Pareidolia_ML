"""
Configuración y constantes del proyecto Pareidolia ML
"""

import os

# Dimensiones de imagen
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_WIDTH_XC = 299
IMAGE_HEIGHT_XC = 299
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

# Hiperparámetros de entrenamiento
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.15

# Categorías
CATEGORIES = ['sin-cara', 'cara']
CATEGORY_TO_ID = {cat: i for i, cat in enumerate(CATEGORIES)}

# Rutas (relativas a la raíz del proyecto)
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
TRAIN_PATH = os.path.join(ROOT_PATH, 'data/train/')
TEST_PATH = os.path.join(ROOT_PATH, 'data/test/')
PREDICT_PATH = os.path.join(ROOT_PATH, 'data/predictions/')
GRADCAM_PATH = os.path.join(ROOT_PATH, 'data/predictions/grad_cam/')
MODEL_PATH = os.path.join(ROOT_PATH, 'src/model/')
PRODUCTION_MODEL_PATH = os.path.join(MODEL_PATH, 'production/')
DATA_NPZ_PATH = os.path.join(ROOT_PATH, 'data/data.npz')
DATA_GRAY_NPZ_PATH = os.path.join(ROOT_PATH, 'data/data_gray.npz')
DATA_AUG_NPZ_PATH = os.path.join(ROOT_PATH, 'data/data_aug.npz')

# Configuración de modelos
MODEL_CONFIG = {
    'EfficientNetB0': {
        'input_shape': (224, 224, 3),
        'weights': 'imagenet',
        'dense_units': 128,
        'dropout': 0.4,
    },
    'ResNet50': {
        'input_shape': (224, 224, 3),
        'weights': 'imagenet',
        'dense_units': 128,
        'dropout': 0.4,
    },
    'Xception': {
        'input_shape': (299, 299, 3),
        'weights': 'imagenet',
        'dense_units': 128,
        'dropout': 0.4,
    },
}

# Parámetros de fine-tuning
FINETUNE_LAYERS = 25
FINETUNE_LR = 1e-5
FINETUNE_EPOCHS = 20

# Threshold de predicción
PREDICTION_THRESHOLD = 0.5
