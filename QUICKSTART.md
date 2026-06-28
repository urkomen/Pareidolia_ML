# Guía de Inicio Rápido — Pareidolia ML

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 2. Lanzar la app web

La forma más rápida de usar el proyecto:

```bash
streamlit run src/utils/streamlit.py
```

Abre el navegador en `http://localhost:8501`. Desde la página **Home** puedes leer
qué es la pareidolia y acceder al detector con un clic.

En la página de **Predicción**:

1. Sube una imagen (JPG/PNG) o usa la cámara web
2. Selecciona el modelo en la barra lateral *(recomendado: Augmented + Fine-tuned)*
3. Ajusta el umbral si es necesario (por defecto 0.5)
4. Pulsa **Ejecutar predicción**

## 3. Usar el modelo desde código

```python
import tensorflow as tf
from src.utils import predict_single_image

# Cargar modelo de producción
model = tf.keras.models.load_model('src/model/production/Xception_augmented_finetuned.keras')

# Predicción en una imagen
result = predict_single_image(model, 'ruta/imagen.jpg', size=(299, 299))
print(f"Predicción: {result['label']}")
print(f"Confianza:  {result['confidence']:.2%}")

# Predicciones en lote
from src.utils import batch_predict
results = batch_predict(model, ['img1.jpg', 'img2.jpg'], size=(299, 299))
```

## 4. Entrenar desde cero

Ejecuta los notebooks en orden desde `src/notebooks/`:

```
01_data_preparation.ipynb   → Carga y serializa datos en data/data.npz
02_model_comparison.ipynb   → Compara EfficientNet, ResNet50 y Xception
03_enhance_models.ipynb     → Fine-tuning, augmentation y combinaciones
04_predictions.ipynb        → Evaluación final y Grad-CAM
```

O desde código:

```python
from src.utils import load_data_npz, build_xception, train_model, evaluate_model

X_train, y_train, X_test, y_test = load_data_npz('data/data.npz')
model = build_xception()
history = train_model(model, X_train, y_train, validation_split=0.15, epochs=30)
results = evaluate_model(model, X_test, y_test)
```

## Archivos clave

| Ruta                                                        | Descripción                                       |
| ----------------------------------------------------------- | -------------------------------------------------- |
| `src/model/production/Xception_augmented_finetuned.keras` | Modelo recomendado                                 |
| `src/utils/streamlit.py`                                  | App web — Home                                    |
| `src/utils/pages/1_Prediccion.py`                         | App web — Detector                                |
| `src/utils/constants.py`                                  | Configuración (rutas, tamaños, hiperparámetros) |
| `memoria.ipynb`                                           | Resumen del proyecto y resultados                  |
| `Pareidolia.md`                                           | Documentación técnica completa                   |

## Personalización

Edita `src/utils/constants.py` para cambiar:

- `IMAGE_WIDTH_XC`, `IMAGE_HEIGHT_XC` — tamaño de entrada para Xception (299×299)
- `BATCH_SIZE`, `EPOCHS` — parámetros de entrenamiento
- `PREDICTION_THRESHOLD` — umbral por defecto de clasificación
- `TRAIN_PATH`, `TEST_PATH` — rutas al dataset

## Troubleshooting

**Módulo no encontrado**

```bash
# Ejecuta siempre desde la raíz del proyecto
cd Pareidolia_ML
streamlit run src/utils/streamlit.py
```

**GPU no disponible**

```python
import tensorflow as tf
print("GPUs disponibles:", len(tf.config.list_physical_devices('GPU')))
# El proyecto funciona en CPU también, pero más lento
```

**Error en Grad-CAM**
Asegúrate de usar el modelo Xception — la capa `block14_sepconv2_act` es específica
de esa arquitectura y no existe en EfficientNet ni ResNet.
