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
   → Carga, normaliza y guarda datos en .npz

2. src/notebooks/02_model_comparison.ipynb
   → Compara 3 arquitecturas (EfficientNet, ResNet, Xception)

3. src/notebooks/03_fine_tuning.ipynb
   → Mejora el mejor modelo con fine-tuning

4. src/notebooks/04_predictions.ipynb
   → Valida en dataset de test
```

## 💡 Ejemplo Rápido

```python
import sys
sys.path.append('src')

from utils import (
    load_and_prepare_data,
    build_xception,
    train_model,
    evaluate_model
)

# 1. Cargar datos
X_train, X_test, y_train, y_test = load_and_prepare_data()

# 2. Construir modelo
model = build_xception()

# 3. Entrenar
history = train_model(model, X_train, y_train, epochs=30)

# 4. Evaluar
results = evaluate_model(model, X_test, y_test, model_name="Xception")
```

## 📁 Archivos Importantes

### Módulos (src/utils/)

- `constants.py` - Configuración
- `data_loader.py` - Carga de datos
- `model_builder.py` - Construcción de modelos
- `training.py` - Entrenamiento
- `evaluation.py` - Evaluación
- `prediction.py` - Predicciones

### Notebooks (src/notebooks/)

- `01_data_preparation.ipynb`
- `02_model_comparison.ipynb`
- `03_fine_tuning.ipynb`
- `04_predictions.ipynb`

### Documentación

- `src/memoria.ipynb` - Resumen del proyecto
- `src/README.md` - Documentación técnica

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

Consulta `src/memoria.ipynb` para un resumen completo del proyecto.
