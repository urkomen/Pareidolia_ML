"""
Módulo para construir y compilar modelos de transfer learning
"""

from tensorflow.keras.applications import EfficientNetB0, ResNet50, Xception, InceptionV3, VGG16
from tensorflow.keras import layers, models, optimizers
from sklearn.utils import shuffle
import numpy as np
from .constants import MODEL_CONFIG

def build_model(backbone_fn, input_shape=(224, 224, 3), weights='imagenet', 
                dense_units=128, dropout=0.4, learning_rate=1e-4):
    """
    Construye un modelo de transfer learning con un backbone preentrenado.
    
    Args:
        backbone_fn: Función del backbone (ej: EfficientNetB0)
        input_shape (tuple): Forma de entrada (height, width, channels)
        weights (str): Pesos preentrenados a usar
        dense_units (int): Unidades en la capa densa
        dropout (float): Tasa de dropout
        learning_rate (float): Learning rate del optimizer
    
    Returns:
        model: Modelo de Keras compilado
    """
    # Cargar backbone preentrenado
    base = backbone_fn(include_top=False, weights=weights, input_shape=input_shape)
    base.trainable = False  # Congelar el backbone inicialmente
    
    # Construir modelo
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')  # Salida binaria
    ])
    
    # Compilar
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_efficient_net_b0(input_shape=(224, 224, 3), learning_rate=1e-4):
    """Construye modelo EfficientNetB0"""
    return build_model(EfficientNetB0, input_shape=input_shape, learning_rate=learning_rate)


def build_resnet50(input_shape=(224, 224, 3), learning_rate=1e-4):
    """Construye modelo ResNet50"""
    return build_model(ResNet50, input_shape=input_shape, learning_rate=learning_rate)


def build_xception(input_shape=(299, 299, 3), learning_rate=1e-4):
    """Construye modelo Xception"""
    return build_model(Xception, input_shape=input_shape, learning_rate=learning_rate)


def build_inception_v3(input_shape=(299, 299, 3), learning_rate=1e-4):
    """Construye modelo InceptionV3"""
    return build_model(InceptionV3, input_shape=input_shape, learning_rate=learning_rate)


def build_vgg16(input_shape=(224, 224, 3), learning_rate=1e-4):
    """Construye modelo VGG16 (menos recomendado por tamaño)"""
    return build_model(VGG16, input_shape=input_shape, learning_rate=learning_rate)


def freeze_backbone(model):
    """Congela todas las capas del backbone"""
    # El primer elemento es el backbone
    model.layers[0].trainable = False
    return model


def unfreeze_backbone_layers(model, num_layers=25):
    """
    Descongela las últimas N capas del backbone para fine-tuning.
    
    Args:
        model: Modelo Sequential
        num_layers (int): Número de capas a descongelar desde el final
    
    Returns:
        model: Modelo modificado
    """
    # El primer elemento es el backbone
    base = model.layers[0]
    
    # Congelar todas
    for layer in base.layers:
        layer.trainable = False
    
    # Descongelar las últimas N
    for layer in base.layers[-num_layers:]:
        layer.trainable = True
    
    print(f"Capas entrenables en backbone: {sum(l.trainable for l in base.layers)}/{len(base.layers)}")
    
    return model


def augmentar_dataset(X_train, y_train, datagen, clase_minoritaria=1, random_state=42):
    '''
    Aumenta el dataset aplicando transformaciones sintéticas sobre la clase minoritaria
    para equilibrar la distribución de clases en el conjunto de entrenamiento.

    Args:
        X_train : np.ndarray
            Array de imágenes de entrenamiento con shape (N, H, W, C).
        y_train : np.ndarray
            Array de etiquetas correspondientes a X_train.
        clase_minoritaria : int, opcional
            Índice de la clase sobre la que se aplica el augmentation (por defecto 1 = cara).
        random_state : int, opcional
            Semilla para la mezcla aleatoria del dataset resultante (por defecto 42).

    Returns:
        X_train_aug : np.ndarray
            Dataset aumentado con las imágenes originales más las sintéticas generadas.
        y_train_aug : np.ndarray
            Etiquetas correspondientes al dataset aumentado.

    Ejemplo
    -------
    >>> X_train_aug, y_train_aug = augmentar_dataset(X_train_xception, y_train, clase_minoritaria=1)
    '''
    
    idx_minoritaria = np.where(y_train == clase_minoritaria)[0]
    X_minoritaria = X_train[idx_minoritaria]
    y_minoritaria = y_train[idx_minoritaria]

    X_aug, y_aug = [], []
    for X_batch, y_batch in datagen.flow(X_minoritaria, y_minoritaria, batch_size=len(X_minoritaria)):
        X_aug.append(X_batch)
        y_aug.append(y_batch)
        break

    X_train_aug = np.concatenate([X_train, X_aug[0]])
    y_train_aug = np.concatenate([y_train, y_aug[0]])
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=random_state)

    return X_train_aug, y_train_aug


def get_model_summary(model):
    """Obtiene resumen del modelo"""
    return model.summary()


def get_trainable_layers_count(model):
    """Retorna número de capas entrenables"""
    return sum(1 for layer in model.layers if layer.trainable)


def get_total_layers_count(model):
    """Retorna número total de capas"""
    return len(model.layers)
