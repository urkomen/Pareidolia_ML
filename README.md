# Pareidolia ML - Estructura del Proyecto

## 📂 Descripción General

Este proyecto implementa un clasificador de imágenes para detectar pareidolia (ilusión óptica de caras) usando Transfer Learning con modelos preentrenados.

## 📋 Estructura de Carpetas

```
src/
├── utils/                          # Módulos reutilizables
│   ├── __init__.py                # Inicialización del paquete
│   ├── constants.py               # Configuración y constantes
│   ├── data_loader.py             # Carga y procesamiento de datos
│   ├── model_builder.py           # Construcción de modelos
│   ├── training.py                # Funciones de entrenamiento
│   ├── evaluation.py              # Evaluación y visualización
│   └── prediction.py              # Predicciones e inferencia
│
├── data/                           # Datasets procesados
│   ├── data.npz                   # Datos normalizados (formato comprimido)
│   └── data_gray.npz              # Datos en escala de grises (alternativa)
│
├── notebooks/                      # Notebooks ejecutables
│   ├── 01_data_preparation.ipynb  # Carga y preparación de datos
│   ├── 02_model_comparison.ipynb  # Comparación de arquitecturas
│   ├── 03_fine_tuning.ipynb       # Fine-tuning del mejor modelo
│   └── 04_predictions.ipynb       # Predicciones y validación
│
├── model/                          # Modelos entrenados
│   ├── production/                # Modelo final para producción
│   │   └── best_model.keras       # Modelo optimizado
│   └── *.keras                    # Modelos experimentales
│
├── memoria.ipynb                   # Documentación del proyecto
└── README.md                       # Este archivo
```

## 🚀 Cómo Usar

### 1. Instalación de dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar notebooks en orden

```bash
# Orden recomendado:
jupyter notebook notebooks/01_data_preparation.ipynb
jupyter notebook notebooks/02_model_comparison.ipynb
jupyter notebook notebooks/03_fine_tuning.ipynb
jupyter notebook notebooks/04_predictions.ipynb
```

### 3. Usar módulos en tu código

```python
import sys
sys.path.append('../../src')

from utils import (
    load_and_prepare_data,
    build_xception,
    train_model,
    evaluate_model,
    predict_and_visualize
)

# Cargar datos
X_train, X_test, y_train, y_test = load_and_prepare_data()

# Construir modelo
model = build_xception()

# Entrenar
history = train_model(model, X_train, y_train)

# Evaluar
results = evaluate_model(model, X_test, y_test)
```

## 📦 Módulos Disponibles

### `constants.py`
- Configuración centralizada
- Rutas de proyecto
- Hiperparámetros

### `data_loader.py`
- `read_data()` - Carga imágenes desde carpeta
- `load_and_prepare_data()` - Pipeline completo
- `save_data_npz()` - Guardar en formato comprimido
- `load_data_npz()` - Cargar desde .npz
- `preprocess_image()` - Preprocesar imagen individual

### `model_builder.py`
- `build_model()` - Constructor genérico
- `build_efficient_net_b0()` - EfficientNet
- `build_resnet50()` - ResNet50
- `build_xception()` - Xception (mejor modelo)
- `unfreeze_backbone_layers()` - Para fine-tuning

### `training.py`
- `train_model()` - Entrenamiento estándar
- `train_model_custom()` - Con validation set personalizado
- `train_with_augmentation()` - Con data augmentation

### `evaluation.py`
- `evaluate_model()` - Evaluación completa
- `plot_confusion_matrix()` - Matriz de confusión
- `plot_learning_curves()` - Curvas de aprendizaje
- `plot_roc_curve()` - Curva ROC
- `find_optimal_threshold()` - Búsqueda de threshold

### `prediction.py`
- `predict_single_image()` - Predicción en imagen
- `batch_predict()` - Predicciones en lote
- `visualize_predictions()` - Grilla de resultados
- `get_predictions_summary()` - Estadísticas

## 📊 Resultados del Proyecto

| Modelo | Accuracy | AUC-ROC | Entrada |
|--------|----------|---------|---------|
| EfficientNetB0 | [Insertar] | [Insertar] | 224×224 |
| ResNet50 | [Insertar] | [Insertar] | 224×224 |
| **Xception** | **[Insertar]** | **[Insertar]** | **299×299** |
| Xception Fine-tuned | **[Insertar]** | **[Insertar]** | **299×299** |

## 📝 Documentación

Consulta `memoria.ipynb` para:
- Descripción completa del proyecto
- Detalles de cada paso
- Código de ejemplo
- Próximos pasos

## 🔧 Configuración

Editar `utils/constants.py` para personalizar:
- Dimensiones de imagen
- Rutas de datos
- Hiperparámetros de entrenamiento
- Parámetros de fine-tuning

## 🎯 Uso en Producción

```python
import tensorflow as tf
from utils import predict_and_visualize

# Cargar modelo
model = tf.keras.models.load_model('model/production/best_model.keras')

# Predecir
result = predict_and_visualize(
    model,
    'ruta/a/imagen.jpg',
    size=(299, 299),
    threshold=0.5
)

print(f"Predicción: {result['label']}")
print(f"Confianza: {result['confidence']:.2%}")
```

## 📚 Referencias

- [Keras Transfer Learning](https://keras.io/guides/transfer_learning/)
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [ResNet: Deep Residual Learning](https://arxiv.org/abs/1512.03385)

## ⚖️ Licencia

[Especificar licencia]

## 👤 Autor

[Tu nombre/información]

---

**Nota:** Para preservar el notebook original `modelo.ipynb`, se ha extraído todo el código a módulos reutilizables en `utils/` y notebooks organizados en `notebooks/`.
