# Pareidolia ML - Estructura del Proyecto

## 📂 Descripción General

Este proyecto implementa un clasificador de imágenes para detectar pareidolia (ilusión óptica de caras) usando Transfer Learning con modelos preentrenados.

## 📋 Estructura de Carpetas

```
Pareidolia_ML/
├── data/                           # Datasets
│   ├── train/                      # Imágenes de entrenamiento
│   │   ├── cara/
│   │   └── sin-cara/
│   ├── test/                       # Imágenes de test
│   │   ├── cara/
│   │   └── sin-cara/
│   ├── predictions/                # Predicciones y visualizaciones
│   │   └── grad_cam/
│   ├── data.npz                    # Datos normalizados (comprimido)
│   ├── data_gray.npz               # Datos en escala de grises
│   └── data_aug.npz                # Datos aumentados
│
├── src/
│   ├── utils/                      # Módulos reutilizables
│   │   ├── __init__.py
│   │   ├── constants.py            # Configuración y constantes
│   │   ├── data_loader.py          # Carga y procesamiento
│   │   ├── model_builder.py        # Construcción de modelos
│   │   ├── training.py             # Funciones de entrenamiento
│   │   ├── evaluation.py           # Evaluación y visualización
│   │   └── prediction.py           # Predicciones e inferencia
│   │
│   ├── model/                      # Modelos entrenados
│   │   ├── Xception.keras          # Modelo base
│   │   ├── Xception_finetuned.keras
│   │   ├── Xception_augmented.keras
│   │   ├── Xception_augmented_finetuned.keras  # ⭐ Recomendado
│   │   └── production/             # Modelos para producción
│   │
│   └── notebooks/                  # Notebooks ejecutables
│       ├── 01_data_preparation.ipynb
│       ├── 02_model_comparison.ipynb
│       ├── 03_enhance_models.ipynb
│       └── 04_predictions.ipynb
│
├── resources/
│   └── img/                        # Imágenes y recursos visuales
│
├── memoria.ipynb                   # Resumen del proyecto
├── Pareidolia.md                   # Documentación técnica
├── QUICKSTART.md                   # Guía de inicio rápido
├── README.md                       # Este archivo
└── requirements.txt
```

## 🚀 Cómo Usar

### 1. Instalación de dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar notebooks en orden

```bash
# Ubicación: src/notebooks/

# 1. Preparación de datos
jupyter notebook src/notebooks/01_data_preparation.ipynb

# 2. Comparación de modelos
jupyter notebook src/notebooks/02_model_comparison.ipynb
odelo preentrenado

```python
import tensorflow as tf
from src.utils import predict_single_image, batch_predict

# Cargar modelo recomendado de producción
model = tf.keras.models.load_model('src/model/Xception_augmented_finetuned.keras')

# Predicción en una imagen
result = predict_single_image(model, 'ruta/imagen.jpg')
print(f"Probabilidad de pareidolia: {result:.2%}")

# Predicciones en lote
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch_results = batch_predict(model, image_paths)
```

### 4. Usar módulos en tu código

```python
import sys
sys.path.append('src')

from utils import (
    load_data_npz,
    build_xception,
    train_model,
    evaluate_model
)

# Cargar datos
X_train, y_train, X_test, y_test = load_data_npz('data/data.npz')

# Construir modelo
model = build_xception()

# Entrenar
history = train_model(model, X_train, y_train, epochs=30
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
- `build_resnet50()` - ResNet50 individual
- `batch_predict()` - Predicciones en lote
- `visualize_predictions()` - Grilla de resultados
- `get_predictions_summary()` - Estadísticas agregad

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

## 📊 Modelos Disponibles

| Modelo                    | Archivo                                          | Descripción                | Recomendado  |
| ------------------------- | ------------------------------------------------ | --------------------------- | ------------ |
| Xception Base             | `Xception.keras`                               | Backbone congelado          | -            |
| Xception Fine-tuned       | `Xception_finetuned.keras`                     | Con fine-tuning             | -            |
| Xception Augmented        | `Xception_augmented.keras`                     | Con data augmentation       | -            |
| **Xception Aug+FT** | **`Xception_augmented_finetuned.keras`** | **Mejor rendimiento** | **✓** |

Todos los modelos: Disponible

- **`memoria.ipynb`** - Resumen del proyecto, arquitectura y resultados
- **`Pareidolia.md`** - Documentación técnica completa con análisis detallado
- **`QUICKSTART.md`** - Guía de inicio rápido con ejemplos
- **`README.md`** - Este archivo
  Consulta `memoria.ipynb` para:
- Descripción completa del proyecto
- Detalles de cada paso
- Código de ejemplo
- Próximos pasos

## 🔧 Configuración

Editar `utils/constants.py` para personalizar:

- Dimenssrc/utils/constants.py` para personalizar:
- `IMAGE_WIDTH`, `IMAGE_HEIGHT` - Dimensiones de entrada (para Xception: 299×299)
- `BATCH_SIZE`, `EPOCHS` - Parámetros de entrenamiento
- `TRAIN_PATH`, `TEST_PATH` - Rutas a datos
- Modelos y hiperparámetros adicionales

## 🎯 Uso en Producción

```python
import tensorflow as tf
from utils import predict_and_visualize
src.utils import predict_single_image

# Cargar modelo recomendado
model = tf.keras.models.load_model('src/model/Xception_augmented_finetuned.keras')

# Predicción
probability = predict_single_image(model, 'ruta/imagen.jpg')

# Resultado
threshold = 0.5
label = 'cara (pareidolia)' if probability > threshold else 'sin-cara'
confidence = max(probability, 1 - probability)

print(f"Predicción: {label}")
print(f"Confianza: {confidence

## 📚 Referencias

- [Keras Transfer Learning](https://keras.io/guides/transfer_learning/)
- [Xception: Deep Learning Guide](https://keras.io/guides/transfer_learning/)
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Data Augmentation with TensorFlow](https://www.tensorflow.org/tutorials/images/data_augmentation

## 👤 Autor

Urko Menendez

---
Para empezar rápidamente:** consulta [QUICKSTART.md](QUICKSTART.md)  
**Para documentación técnica:** consulta [Pareidolia.md](Pareidolia.md)  
**Para resumen del proyecto:** abre [memoria.ipynb](memoria.ipynb)
**Nota:** Para preservar el notebook original `modelo.ipynb`, se ha extraído todo el código a módulos reutilizables en `utils/` y notebooks organizados en `notebooks/`.
```
