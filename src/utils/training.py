"""
Módulo para entrenar modelos
"""

from tensorflow.keras import callbacks
import tensorflow as tf


def train_model(model, X_train, y_train, epochs=30, batch_size=32, 
                validation_split=0.15, patience=7, verbose=1):
    """
    Entrena un modelo con early stopping y learning rate reduction.
    
    Args:
        model: Modelo de Keras
        X_train, y_train: Datos de entrenamiento
        epochs (int): Número máximo de épocas
        batch_size (int): Tamaño del batch
        validation_split (float): Proporción para validación
        patience (int): Paciencia para early stopping
        verbose (int): Nivel de verbosidad
    
    Returns:
        history: Objeto History de Keras con métricas de entrenamiento
    """
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience // 2,
        verbose=verbose,
        min_lr=1e-7
    )
    
    cb = [early_stop, reduce_lr]
    
    # Entrenar
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=verbose
    )
    
    return history


def train_model_custom(model, X_train, y_train, X_val, y_val, 
                       epochs=30, batch_size=32, patience=7, verbose=1):
    """
    Entrena con un validation set personalizado.
    
    Args:
        model: Modelo de Keras
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        epochs (int): Número máximo de épocas
        batch_size (int): Tamaño del batch
        patience (int): Paciencia para early stopping
        verbose (int): Nivel de verbosidad
    
    Returns:
        history: Objeto History de Keras
    """
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience // 2,
        verbose=verbose,
        min_lr=1e-7
    )
    
    cb = [early_stop, reduce_lr]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=verbose
    )
    
    return history


def train_with_augmentation(model, X_train, y_train, datagen, 
                            epochs=30, batch_size=32, 
                            validation_split=0.15, patience=7, verbose=1):
    """
    Entrena usando data augmentation en tiempo real.
    
    Args:
        model: Modelo de Keras
        X_train, y_train: Datos de entrenamiento
        datagen: ImageDataGenerator configurado
        epochs (int): Número máximo de épocas
        batch_size (int): Tamaño del batch
        validation_split (float): Proporción para validación
        patience (int): Paciencia para early stopping
        verbose (int): Nivel de verbosidad
    
    Returns:
        history: Objeto History de Keras
    """
    
    # Dividir datos
    num_samples = len(X_train)
    num_train = int(num_samples * (1 - validation_split))
    
    X_train_split = X_train[:num_train]
    y_train_split = y_train[:num_train]
    X_val = X_train[num_train:]
    y_val = y_train[num_train:]
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=patience // 2,
        verbose=verbose,
        min_lr=1e-7
    )
    
    cb = [early_stop, reduce_lr]
    
    history = model.fit(
        datagen.flow(X_train_split, y_train_split, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=cb,
        verbose=verbose
    )
    
    return history
