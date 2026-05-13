"""
Módulo para evaluación y visualización de modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_fscore_support, accuracy_score
)


def evaluate_model(model, X_test, y_test, threshold=0.5, model_name="Model"):
    """
    Evalúa un modelo y calcula métricas.
    
    Args:
        model: Modelo entrenado
        X_test, y_test: Datos de test
        threshold (float): Threshold para clasificación binaria
        model_name (str): Nombre del modelo
    
    Returns:
        dict: Diccionario con resultados y métricas
    """
    
    # Predicciones
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Reporte detallado
    report = classification_report(y_test, y_pred, 
                                   target_names=['sin_cara', 'cara'],
                                   output_dict=True)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, 
                                target_names=['sin_cara', 'cara']))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'auc': auc,
        'y_prob': y_prob,
        'y_pred': y_pred,
        'report': report,
        'confusion_matrix': cm,
        'y_test': y_test
    }


def plot_confusion_matrix(cm, model_name, ax=None):
    """
    Grafica la matriz de confusión.
    
    Args:
        cm: Matriz de confusión
        model_name (str): Nombre del modelo
        ax: Eje de matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(f'{model_name} — Matriz de confusión')
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', 
                   fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['sin_cara', 'cara'])
    ax.set_yticklabels(['sin_cara', 'cara'])
    
    return ax


def plot_learning_curves(history, model_name, ax=None):
    """
    Grafica las curvas de aprendizaje.
    
    Args:
        history: Objeto History de Keras
        model_name (str): Nombre del modelo
        ax: Eje de matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history.history['accuracy'], label='Train accuracy', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Val accuracy', linewidth=2)
    ax.set_title(f'{model_name} — Curvas de aprendizaje')
    ax.set_xlabel('Época')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_roc_curve(y_test, y_prob, auc, model_name, ax=None):
    """
    Grafica la curva ROC.
    
    Args:
        y_test: Etiquetas verdaderas
        y_prob: Probabilidades predichas
        auc (float): Valor de AUC
        model_name (str): Nombre del modelo
        ax: Eje de matplotlib
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return ax


def compare_models_roc(results_list, y_test):
    """
    Compara múltiples modelos en una sola curva ROC.
    
    Args:
        results_list: Lista de diccionarios con resultados
        y_test: Etiquetas verdaderas
    """
    plt.figure(figsize=(10, 8))
    
    for result in results_list:
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        plt.plot(fpr, tpr, 
                label=f"{result['model_name']} (AUC={result['auc']:.3f})", 
                linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparación de Modelos - Curvas ROC')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def summary_table(results_list):
    """
    Crea una tabla resumen con resultados de múltiples modelos.
    
    Args:
        results_list: Lista de diccionarios con resultados
    
    Returns:
        DataFrame: Tabla con resumen de modelos
    """
    summary = pd.DataFrame([{
        'Modelo': r['model_name'],
        'Accuracy': f"{r['accuracy']:.4f}",
        'AUC-ROC': f"{r['auc']:.4f}",
    } for r in results_list]).sort_values('AUC-ROC', ascending=False)
    
    print('\n' + '='*60)
    print('RESUMEN DE MODELOS')
    print('='*60)
    print(summary.to_string(index=False))
    
    return summary


def find_optimal_threshold(y_test, y_prob, thresholds=None):
    """
    Encuentra el threshold óptimo que maximiza F1.
    
    Args:
        y_test: Etiquetas verdaderas
        y_prob: Probabilidades predichas
        thresholds: Lista de thresholds a probar
    
    Returns:
        dict: Información del threshold óptimo
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.9, 0.01)
    
    results = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_t, 
                                                       average='binary', 
                                                       zero_division=0)
        acc = accuracy_score(y_test, y_pred_t)
        results.append({
            'Threshold': f'{t:.2f}',
            'Precision': f'{p:.4f}',
            'Recall': f'{r:.4f}',
            'F1': f'{f1:.4f}',
            'Accuracy': f'{acc:.4f}',
            'f1_numeric': f1
        })
    
    df_results = pd.DataFrame(results)
    best_idx = df_results['f1_numeric'].idxmax()
    
    print('\n' + '='*60)
    print('BÚSQUEDA DE THRESHOLD ÓPTIMO')
    print('='*60)
    print(df_results.drop('f1_numeric', axis=1).to_string(index=False))
    print(f"\n✓ Threshold óptimo: {results[best_idx]['Threshold']}")
    
    return results[best_idx]


def plot_probability_distribution(y_prob, y_test, model_name):
    """
    Grafica la distribución de probabilidades predichas.
    
    Args:
        y_prob: Probabilidades predichas
        y_test: Etiquetas verdaderas
        model_name (str): Nombre del modelo
    """
    plt.figure(figsize=(12, 4))
    
    # Histograma general
    plt.subplot(1, 2, 1)
    plt.hist(y_prob, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Frecuencia')
    plt.title(f'{model_name} — Distribución de probabilidades')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Por clase
    plt.subplot(1, 2, 2)
    y_prob_sin_cara = y_prob[y_test == 0]
    y_prob_cara = y_prob[y_test == 1]
    plt.hist(y_prob_sin_cara, bins=15, alpha=0.6, label='sin_cara', edgecolor='black')
    plt.hist(y_prob_cara, bins=15, alpha=0.6, label='cara', edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Probabilidad predicha')
    plt.ylabel('Frecuencia')
    plt.title('Distribución por clase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"\nEstadísticas de probabilidades ({model_name}):")
    print(f"  Min:    {y_prob.min():.4f}")
    print(f"  Max:    {y_prob.max():.4f}")
    print(f"  Media:  {y_prob.mean():.4f}")
    print(f"  Std:    {y_prob.std():.4f}")
    print(f"\nDistribución y_test → sin_cara: {(y_test==0).sum()} | cara: {(y_test==1).sum()}")
    
    plt.tight_layout()
    plt.show()
