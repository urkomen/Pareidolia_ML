# Pareidolia ML

Clasificador de imágenes para detectar **pareidolia** (percepción de caras en objetos inanimados) usando Transfer Learning con Xception.

## Estructura del Proyecto

```
Pareidolia_ML/
├── data/
│   ├── train/
│   │   ├── cara/               # 416 imágenes con pareidolia
│   │   └── sin-cara/           # 652 imágenes sin pareidolia
│   ├── test/
│   │   ├── cara/
│   │   └── sin-cara/
│   ├── predictions/
│   │   └── grad_cam/           # Visualizaciones Grad-CAM generadas
│   ├── data.npz                # Datos normalizados (comprimido)
│   ├── data_gray.npz
│   └── data_aug.npz
│
├── src/
│   ├── utils/
│   │   ├── constants.py        # Configuración y constantes
│   │   ├── data_loader.py      # Carga y preprocesamiento
│   │   ├── model_builder.py    # Arquitecturas de modelos
│   │   ├── training.py         # Funciones de entrenamiento
│   │   ├── evaluation.py       # Evaluación y métricas
│   │   ├── prediction.py       # Predicciones e inferencia
│   │   ├── streamlit.py        # App web — página Home
│   │   └── pages/
│   │       └── 1_Prediccion.py # App web — página de predicción
│   │
│   ├── model/
│   │   ├── Xception.keras
│   │   ├── Xception_finetuned.keras
│   │   ├── Xception_augmented.keras
│   │   ├── Xception_augmented_finetuned.keras
│   │   └── production/
│   │       └── Xception_augmented_finetuned.keras  # Modelo en producción
│   │
│   └── notebooks/
│       ├── 00_data_analysis.ipynb
│       ├── 01_data_preparation.ipynb
│       ├── 02_model_comparison.ipynb
│       ├── 03_enhance_models.ipynb
│       └── 04_predictions.ipynb
│
├── resources/img/
├── memoria.ipynb               # Resumen del proyecto
├── Pareidolia.md               # Documentación técnica
├── QUICKSTART.md               # Guía de inicio rápido
└── requirements.txt
```

## Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Lanzar la app web

```bash
streamlit run src/utils/streamlit.py
```

### 3. Usar el modelo desde código

```python
import tensorflow as tf
from src.utils import predict_single_image

model = tf.keras.models.load_model('src/model/production/Xception_augmented_finetuned.keras')
result = predict_single_image(model, 'ruta/imagen.jpg', size=(299, 299))
print(f"Predicción: {result['label']}  —  Confianza: {result['confidence']:.2%}")
```

Consulta [QUICKSTART.md](QUICKSTART.md) para más ejemplos.

## App Web (Streamlit)

La app incluye dos páginas:

- **Home** — explicación del fenómeno de pareidolia y cómo usar la app
- **Predicción** — carga de imagen, predicción y visualización Grad-CAM

Funcionalidades:

- Subida de imagen o captura con cámara web
- Selector de modelo y umbral de clasificación ajustable
- Resultado con probabilidad y confianza
- Mapa Grad-CAM (zona de atención del modelo)

## Modelos Disponibles

| Modelo                    | Archivo                                            | Descripción                |
| ------------------------- | -------------------------------------------------- | --------------------------- |
| Xception Base             | `Xception.keras`                                 | Backbone congelado          |
| Xception Fine-tuned       | `Xception_finetuned.keras`                       | Con fine-tuning             |
| Xception Augmented        | `Xception_augmented.keras`                       | Con data augmentation       |
| **Xception Aug+FT** |  `production/Xception_augmented_finetuned.keras` | **Mejor rendimiento** |

## Módulos (`src/utils/`)

| Módulo              | Funciones principales                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------- |
| `constants.py`     | Configuración centralizada, rutas, hiperparámetros                                     |
| `data_loader.py`   | `read_data()`, `load_data_npz()`, `preprocess_image()`                             |
| `model_builder.py` | `build_xception()`, `build_efficient_net_b0()`, `build_resnet50()`                 |
| `training.py`      | `train_model()`, `train_with_augmentation()`                                         |
| `evaluation.py`    | `evaluate_model()`, `plot_confusion_matrix()`, `plot_roc_curve()`                  |
| `prediction.py`    | `predict_single_image()`, `batch_predict()`, `prediccion()`, `compute_gradcam()` |
| `streamlit.py`     | App web — Home                                                                          |

## Referencias

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

## Autor

Urko Menendez
