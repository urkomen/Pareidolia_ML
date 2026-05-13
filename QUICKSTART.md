# Guía de Inicio Rápido - Pareidolia ML

## 🎯 Objetivo

Este proyecto implementa un clasificador de pareidolia (ilusión óptica de caras) usando Transfer Learning.

## 📋 Pasos Iniciales

### 1. Asegúrate de tener las dependencias

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib pillow
```

### 2. Estructura de datos esperada

El proyecto asume que tienes imágenes en:

```
data/
├── train/
│   ├── cara/       # Imágenes con pareidolia
│   └── sin-cara/   # Imágenes sin pareidolia
└── test/
    ├── cara/
    └── sin-cara/
```

### 3. Orden de ejecución

```
1. src/notebooks/01_data_preparation.ipynb
   → Carga, normaliza y guarda datos en data/data.npz

2. src/notebooks/02_model_comparison.ipynb
   → Compara 3 arquitecturas (EfficientNet, ResNet, Xception)
   → Genera: src/model/Xception.keras

3. src/notebooks/03_enhance_models.ipynb
   → Fine-tuning, data augmentation y combinaciones
   → Genera: Xception_finetuned.keras, Xception_augmented.keras, Xception_augmented_finetuned.keras

4. src/notebooks/04_predictions.ipynb
   → Valida en dataset de test
```

## 💡 Ejemplo Rápido

### Opción 1: Usar modelo preentrenado (más rápido)

```python
import tensorflow as tf
from src.utils import predict_single_image, batch_predict

# Cargar modelo recomendado
model = tf.keras.models.load_model('src/model/Xception_augmented_finetuned.keras')

# Predicción en una imagen
result = predict_single_image(model, 'ruta/imagen.jpg')
print(f"Probabilidad de pareidolia: {result:.2%}")

# Predicciones en lote
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = batch_predict(model, image_paths)
```

### Opción 2: Entrenar desde cero (completo)

```python
import sys
sys.path.append('src')

from utils import (
    load_data_npz,
    build_xception,
    train_model,
    evaluate_model
)

# 1. Cargar datos (usa cache si existe)
X_train, y_train, X_test, y_test = load_data_npz('data/data.npz')

# 2. Construir modelo
model = build_xception()

# 3. Entrenar
history = train_model(model, X_train, y_train, 
                      validation_split=0.15, epochs=30)

# 4. Evaluar
results = evaluate_model(model, X_test, y_test, model_name="Xception")
```

## 📁 Archivos Importantes

### Modelos Entrenados (src/model/)

- **`Xception.keras`** - Modelo base (baseline)
- **`Xception_finetuned.keras`** - Con fine-tuning
- **`Xception_augmented.keras`** - Con data augmentation
- **`Xception_augmented_finetuned.keras`** - ⭐ Recomendado para producción

### Módulos (src/utils/)

- `constants.py` - Configuración centralizada
- `data_loader.py` - Carga y preprocesamiento de datos
- `model_builder.py` - Construcción de arquitecturas
- `training.py` - Funciones de entrenamiento
- `evaluation.py` - Evaluación y métricas
- `prediction.py` - Predicciones en imágenes

### Notebooks (src/notebooks/)

- `01_data_preparation.ipynb` - Preparación de datos
- `02_model_comparison.ipynb` - Comparativa de modelos
- `03_enhance_models.ipynb` - Fine-tuning y augmentation
- `04_predictions.ipynb` - Evaluación final y Grad-CAM

### Documentación

- `memoria.ipynb` - Resumen del proyecto
- `Pareidolia.md` - Documentación técnica completa
- `README.md` - Información general

## ⚙️ Personalización

Edita `src/utils/constants.py` para cambiar:

- `IMAGE_WIDTH`, `IMAGE_HEIGHT` - Tamaño de entrada
- `BATCH_SIZE`, `EPOCHS` - Parámetros de entrenamiento
- `FINETUNE_LAYERS`, `FINETUNE_LR` - Parámetros de fine-tuning

## 🔍 Troubleshooting

### Error: "Módulo no encontrado"

```python
# Asegúrate de añadir src al path
import sys
sys.path.append('path/to/src')
```

### Error: "Datos no encontrados"

```python
# Verifica que la estructura de carpetas sea correcta:
# data/train/cara/, data/train/sin-cara/
# data/test/cara/, data/test/sin-cara/
```

### GPU no disponible

```python
# El proyecto funcionará en CPU también, pero será más lento
import tensorflow as tf
print("GPUs disponibles:", len(tf.config.list_physical_devices('GPU')))
```

## 📊 Espacio Requerido

- **Datos sin procesar:** ~[Insertar size] MB
- **Archivo .npz comprimido:** ~[Insertar size] MB
- **Modelos entrenados:** ~[Insertar size] MB

## 🚀 Siguiente Paso

Consulta `memoria.ipynb` para un resumen completo del proyecto, o `Pareidolia.md` para la documentación técnica detallada.
