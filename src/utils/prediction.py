"""
Módulo para realizar predicciones con modelos entrenados
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .constants import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH_XC, IMAGE_HEIGHT_XC, PREDICTION_THRESHOLD, PREDICT_PATH, GRADCAM_PATH
from .data_loader import preprocess_image
import cv2 as cv
import tensorflow as tf
import keras


def predict_single_image(model, image_path, threshold=PREDICTION_THRESHOLD, 
                        size=(IMAGE_WIDTH, IMAGE_HEIGHT), grayscale=False):
    """
    Realiza predicción sobre una imagen individual.
    
    Args:
        model: Modelo entrenado
        image_path (str): Ruta de la imagen
        threshold (float): Threshold de clasificación
        size (tuple): Tamaño de la imagen
        grayscale (bool): Si True, convierte a escala de grises
    
    Returns:
        dict: Diccionario con predicción y confianza
    """
    
    img, arr_input = preprocess_image(image_path, size=size, grayscale=grayscale)
    
    prob = model.predict(arr_input, verbose=0)[0][0]
    label = 'cara' if prob >= threshold else 'sin_cara'
    confianza = prob if prob >= threshold else 1 - prob
    
    return {
        'image': img,
        'label': label,
        'probability': prob,
        'confidence': confianza,
        'threshold': threshold
    }


def predict_and_visualize(model, image_path, threshold=PREDICTION_THRESHOLD, 
                         size=(IMAGE_WIDTH, IMAGE_HEIGHT), grayscale=False):
    """
    Predice y visualiza una imagen.
    
    Args:
        model: Modelo entrenado
        image_path (str): Ruta de la imagen
        threshold (float): Threshold de clasificación
        size (tuple): Tamaño de la imagen
        grayscale (bool): Si True, convierte a escala de grises
    """
    
    result = predict_single_image(model, image_path, threshold, size, grayscale)
    
    color = 'green' if result['label'] == 'cara' else 'red'
    
    plt.figure(figsize=(6, 6))
    plt.imshow(result['image'])
    plt.title(f"{result['label'].upper()}  —  Confianza: {result['confidence']:.2%}", 
             color=color, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\nPredicción: {result['label'].upper()}")
    print(f"Probabilidad: {result['probability']:.4f}")
    print(f"Confianza: {result['confidence']:.2%}")
    
    return result


def batch_predict(model, image_paths, threshold=PREDICTION_THRESHOLD, 
                 size=(IMAGE_WIDTH, IMAGE_HEIGHT), grayscale=False):
    """
    Realiza predicciones en lote.
    
    Args:
        model: Modelo entrenado
        image_paths: Lista de rutas de imágenes
        threshold (float): Threshold de clasificación
        size (tuple): Tamaño de la imagen
        grayscale (bool): Si True, convierte a escala de grises
    
    Returns:
        list: Lista de diccionarios con resultados
    """
    
    results = []
    for image_path in image_paths:
        try:
            result = predict_single_image(model, image_path, threshold, size, grayscale)
            results.append(result)
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            results.append({'error': str(e), 'image_path': image_path})
    
    return results


def predict_array(model, X, threshold=PREDICTION_THRESHOLD):
    """
    Realiza predicción sobre un array de imágenes.
    
    Args:
        model: Modelo entrenado
        X: Array de imágenes (N, H, W, C)
        threshold (float): Threshold de clasificación
    
    Returns:
        tuple: (probabilidades, predicciones)
    """
    
    y_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_prob, y_pred


def get_predictions_summary(model, X, y_test, threshold=PREDICTION_THRESHOLD):
    """
    Obtiene resumen de predicciones en un dataset.
    
    Args:
        model: Modelo entrenado
        X: Array de imágenes
        y_test: Etiquetas verdaderas
        threshold (float): Threshold de clasificación
    
    Returns:
        dict: Resumen de predicciones
    """
    
    y_prob, y_pred = predict_array(model, X, threshold)
    
    correct = (y_pred == y_test).sum()
    total = len(y_test)
    accuracy = correct / total
    
    cara_correct = ((y_pred == 1) & (y_test == 1)).sum()
    cara_total = (y_test == 1).sum()
    cara_accuracy = cara_correct / cara_total if cara_total > 0 else 0
    
    sin_cara_correct = ((y_pred == 0) & (y_test == 0)).sum()
    sin_cara_total = (y_test == 0).sum()
    sin_cara_accuracy = sin_cara_correct / sin_cara_total if sin_cara_total > 0 else 0
    
    return {
        'total_accuracy': accuracy,
        'cara_accuracy': cara_accuracy,
        'sin_cara_accuracy': sin_cara_accuracy,
        'total_samples': total,
        'cara_samples': cara_total,
        'sin_cara_samples': sin_cara_total,
        'y_prob': y_prob,
        'y_pred': y_pred
    }


def visualize_predictions(model, X, y_test, num_samples=9, 
                         threshold=PREDICTION_THRESHOLD, figsize=(12, 12)):
    """
    Visualiza predicciones en una grilla.
    
    Args:
        model: Modelo entrenado
        X: Array de imágenes (normalizado entre 0 y 1)
        y_test: Etiquetas verdaderas
        num_samples (int): Número de muestras a visualizar
        threshold (float): Threshold de clasificación
        figsize (tuple): Tamaño de la figura
    """
    
    indices = np.random.choice(len(X), num_samples, replace=False)
    y_prob = model.predict(X[indices], verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    class_names = ['sin_cara', 'cara']
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Mostrar imagen (desnormalizar si es necesario)
        img = X[idx]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        
        # Colores según corrección
        correct = y_pred[i] == y_test[idx]
        color = 'green' if correct else 'red'
        
        title = f"True: {class_names[y_test[idx]]}\n"
        title += f"Pred: {class_names[y_pred[i]]} ({y_prob[i]:.2%})"
        
        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')
    
    # Ocultar axes no usados
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def predicciones(model, path = PREDICT_PATH):
    for file in os.listdir(path):
        if file.lower().endswith(('.jpeg', '.jpg', '.png')):
            
            filename = f'{file}'
            
            img, label, confianza, color = predecir_cara(model, path + filename, size=(299, 299))
            
            # filepath = os.path.join(path, filename)
            # img = cv2.imread(filepath)
            
            img_rgb = cv.cvtColor(np.array(img), cv.COLOR_BGR2RGB)
            img_resized = cv.resize(img_rgb, (IMAGE_WIDTH_XC, IMAGE_HEIGHT_XC))
            img_array = img_resized / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Generar Grad-CAM
            heatmap = compute_gradcam(
                model,
                img_array,
                last_conv_layer_name="block14_sepconv2_act",
            )

            # Superponer
            result = overlay_heatmap(heatmap, img_rgb)

            cv.imwrite(GRADCAM_PATH + 'gradcam_' + filename, result)
            
            # Mostrar imagen
            plt.figure(figsize=(10, 5))

            # Imagen original
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f'{label}  —  confianza: {confianza:.2%}', color=color, fontsize=14, fontweight='bold')
            plt.axis('off')

            # Grad-CAM
            plt.subplot(1, 2, 2)
            plt.imshow(result)
            plt.title('Grad-CAM', color='#8B008B', fontsize=14, fontweight='bold')
            plt.axis('off')

            plt.tight_layout();



# Hay que retocar el filename, por ahora no funciona bien
def prediccion(model, path):
    '''
    Predicción de una imagen
    '''
    
    img, label, confianza, color = predecir_cara(model, path, size=(299, 299))
    
    # filepath = os.path.join(path, filename)
    # img = cv2.imread(filepath)
    
    img_rgb = cv.cvtColor(np.array(img), cv.COLOR_BGR2RGB)
    img_resized = cv.resize(img_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Generar Grad-CAM
    heatmap = compute_gradcam(
        model,
        img_array,
        last_conv_layer_name="block14_sepconv2_act",
    )

    # Superponer
    result = overlay_heatmap(heatmap, img_rgb)

    cv.imwrite(PREDICT_PATH + 'gradcam_' + filename, result)
    
    # Mostrar imagen
    plt.figure(figsize=(10, 5))

    # Imagen original
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'{label}  —  confianza: {confianza:.2%}', color=color, fontsize=14, fontweight='bold')
    plt.axis('off')

    # Grad-CAM
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title('Grad-CAM', color='#8B008B', fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout();



def compute_gradcam(model, img_array, last_conv_layer_name="block14_sepconv2_act", class_index=None):
    # 1. Backbone: Xception dentro del Sequential
    base_model = model.get_layer("xception")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # 2. Reconstruir la cabeza (las capas después de xception)
    x = base_model.output
    for layer in model.layers[1:]:  # saltamos xception
        x = layer(x)
    top_output = x  # salida final (igual que model.output)

    # 3. Modelo que devuelve: [activaciones conv, salida final]
    grad_model = keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, top_output],
    )

    # 4. Forward + backward en el mismo grafo
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)

        conv_outputs, preds = grad_model(inputs)

        if class_index is None:
            class_index = tf.argmax(preds[0])

        class_channel = preds[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradientes None: la capa no está conectada a la salida.")

    # 5. Gradiente medio por canal
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Ponderar mapas de activación
    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)  # (H, W)

    # 7. Normalizar
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap



def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """
    Superpone el heatmap sobre la imagen original.
    original_img: imagen en formato uint8 (H, W, 3)
    """
    heatmap = cv.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    superimposed = cv.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

    return superimposed



def predecir_cara(model, image_path, size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    img, arr_input = preprocess_image(image_path, size)

    prob = model.predict(arr_input, verbose=0)[0][0]
    label = 'CARA' if prob >= 0.5 else 'SIN CARA'
    confianza = prob if prob >= 0.5 else 1 - prob
    color = 'green' if label == 'CARA' else 'red'
    
    return img, label, confianza, color