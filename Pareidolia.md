# Pareidolia_ML — Detección de Pareidolia mediante Transfer Learning

---

## 1. Introducción a la pareidolia

La pareidolia es el fenómeno cognitivo por el cual el cerebro humano percibe patrones familiares —especialmente rostros— en estímulos ambiguos o aleatorios: nubes, manchas, cortezas de árbol, objetos cotidianos. No se trata de un error perceptivo, sino de un mecanismo adaptativo profundamente arraigado en la evolución humana. La capacidad de reconocer caras con rapidez y en condiciones de baja información visual ha supuesto una ventaja de supervivencia a lo largo de millones de años.

Desde el punto de vista computacional, la pareidolia plantea un problema de clasificación binaria especialmente interesante: ¿puede una red neuronal aprender a detectar aquello que el cerebro humano percibe como una cara, incluso cuando no lo es? Las imágenes de pareidolia comparten rasgos estructurales con los rostros reales —simetría, disposición de elementos que recuerdan ojos, nariz y boca, contraste local— pero sin ninguno de los componentes anatómicos que definen un rostro humano auténtico.

Este proyecto aborda exactamente ese reto: entrenar un clasificador capaz de distinguir entre imágenes donde se percibe una cara (pareidolia) e imágenes donde no existe dicha percepción.

---

## 2. Obtención, análisis y tratamiento del dataset de imágenes

### Dataset

El dataset fue recopilado manualmente a partir de diferentes fuentes de internet (google images, bing...), seleccionando imágenes de objetos cotidianos donde se aprecia la forma de una cara. El conjunto final quedó compuesto por:

* **416 imágenes de clase `cara`** : objetos, superficies o elementos naturales en los que se percibe un rostro.
* **652 imágenes de clase `sin-cara`** : imágenes sin ninguna forma recognoscible de cara.

El dataset presenta un cierto desbalanceo entre clases (aproximadamente 37% / 63%), lo que fue tenido en cuenta en fases posteriores de entrenamiento. Sin embargo, también se asemeja con la realidad donde no vemos caras en todos los objetos todo el rato, sino ocasionadamente.


### Nomenclatura y estructura de ficheros

Las imágenes fueron renombradas con una nomenclatura estandarizada mediante funciones propias (`renombrar_imagenes_cara`, `renombrar_imagenes_sincara`), que aplican el formato `cara_XXX.ext` y `sin-cara_XXX.ext` respectivamente, con comprobaciones previas para evitar colisiones en caso de ejecuciones múltiples. En una tercera carpeta 'predictions' añadimos alguna foto par poner a prueba los modelos entrenados. La estructura de directorios quedó organizada en:

```
data/
├── train/
│   ├── cara/
│   └── sin-cara/
├── test/
│   ├── cara/
│   └── sin-cara/
└── predictions/
```

### Análisis exploratorio (EDA)

Se realizó un análisis exploratorio completo con los siguientes bloques:

**Distribución de tamaños y resoluciones.** Las imágenes presentaban una alta variabilidad en resolución, por lo que fue necesario aplicar un resize uniforme a 224×224 píxeles (299×299 para Xception). Se analizaron histogramas de anchos y altos, y se calcularon media y moda de las dimensiones.

**Color medio por imagen.** Se calculó el color medio en espacio RGB para una muestra de 300 imágenes del conjunto de entrenamiento, confirmando diversidad cromática suficiente en el dataset.

**Visualización aleatoria por clase.** Se implementó la función `show_random_by_class` para mostrar muestras aleatorias de cada categoría en grids de 3×3, permitiendo una inspección visual cualitativa.

**Simetría horizontal.** Se implementó una métrica de simetría basada en la correlación de Pearson entre la mitad izquierda y derecha de cada imagen (convertida a escala de grises). Las caras mostraron una simetría media mayor que las imágenes sin cara, lo que sugiere que la simetría horizontal es una feature relevante para la tarea. Se incluyeron protecciones frente a imágenes constantes que producirían divisiones por cero en el cálculo de correlación.

**Detección de outliers.** Se identificaron imágenes demasiado pequeñas (resolución inferior a 50×50 píxeles) o corruptas mediante la función `find_outliers`, que recorre todos los paths del conjunto de entrenamiento y reporta anomalías.

---

## 3. Entrenamiento de modelos. Elección del mejor tipo de modelo: Xception

### Estrategia de transfer learning

Dado el tamaño moderado del dataset, se optó por transfer learning sobre backbones preentrenados en ImageNet, congelando el backbone completo y entrenando únicamente una cabeza de clasificación. La arquitectura de cabeza empleada fue:

```
GlobalAveragePooling2D → BatchNormalization → Dense(128, relu) → Dropout(0.4) → Dense(1, sigmoid)
```

El entrenamiento utilizó Adam con `lr=1e-4`, `binary_crossentropy` como función de pérdida, y los callbacks `EarlyStopping` (patience=7) y `ReduceLROnPlateau` (patience=3, factor=0.5) para evitar sobreajuste y adaptar la tasa de aprendizaje.

### Backbones comparados

Se evaluaron tres arquitecturas en igualdad de condiciones:

| Modelo         | Input shape | Observaciones                                                                    |
| -------------- | ----------- | -------------------------------------------------------------------------------- |
| EfficientNetB0 | 224×224    | Ligero y eficiente, buen baseline                                                |
| ResNet50       | 224×224    | Robusto, estable en transfer learning                                            |
| Xception       | 299×299    | Basado en separable convolutions, muy bueno captando texturas y patrones locales |

La evaluación se realizó con las métricas Accuracy, F1-score, AUC-ROC y matriz de confusión sobre el conjunto de test. Las curvas ROC permitieron comparar los modelos de forma independiente al threshold de decisión.

### Elección de Xception

Xception obtuvo el mejor AUC-ROC en la comparativa inicial, lo que lo posicionó como el backbone principal para las fases de mejora posteriores. Su capacidad para capturar patrones espaciales locales mediante convoluciones separables en profundidad resulta especialmente adecuada para la tarea de pareidolia, donde la detección de estructuras que evocan ojos, nariz y boca depende de features de textura y forma más que de features de alto nivel semántico.

---

## 4. Mejoras del modelo mediante escala de grises, augmentation y fine-tuning. Visualización con Grad-CAM

### 4.1 Escala de grises

Se exploró la reducción del espacio de features eliminando la información de color, convirtiendo todas las imágenes a escala de grises antes del entrenamiento. Dado que los backbones preentrenados en ImageNet requieren 3 canales de entrada, el canal de gris se replicó tres veces (`np.stack([arr]*3, axis=-1)`) para mantener la compatibilidad.

El pipeline de carga quedó unificado en PIL, manejando explícitamente imágenes en modo paleta con transparencia (`P`, `PA`) mediante conversión intermedia a RGBA antes de pasar a escala de grises, evitando así los warnings de PIL.

### 4.2 Fine-tuning

Se implementó fine-tuning sobre el modelo Xception previamente entrenado, descongelando las últimas 25 capas del backbone y reentrenando con una tasa de aprendizaje muy baja (`lr=1e-5`) para preservar los pesos preentrenados:

```python
# Descongelar las últimas 25 capas del backbone
for layer in base.layers[-25:]:
    layer.trainable = True
```

El entrenamiento de fine-tuning se realizó con batch_size=16 y EarlyStopping (patience=5) para un control más preciso del sobreajuste.

### 4.3 Augmentation

Para reducir el desbalanceo de clases y mejorar la generalización del modelo en imágenes `sin-cara`, se aplicaron augmentations agresivas exclusivamente sobre esa clase usando `ImageDataGenerator`:

* Rotaciones hasta 30°
* Desplazamientos horizontales y verticales (20%)
* Zoom (30%)
* Flip horizontal
* Variaciones de brillo (0.6–1.4)
* Shear (20%)

Esto duplicó el número de imágenes `sin-cara` en el conjunto de entrenamiento, equilibrando mejor la distribución de clases.

### 4.4 Modelo combinado: Augmentation + Fine-tuning

Se entrenó un modelo final combinando ambas mejoras siguiendo el orden correcto: primero entrenar la cabeza con backbone congelado usando el dataset aumentado, y posteriormente aplicar fine-tuning sobre las últimas capas. Este orden es crítico para evitar que gradientes grandes procedentes de una cabeza con pesos aleatorios dañen los pesos preentrenados del backbone.

Los cuatro modelos comparados en la fase final fueron:

| Modelo                           | Descripción                               |
| -------------------------------- | ------------------------------------------ |
| Xception base (escala de grises) | Backbone congelado, imágenes en gris      |
| Xception finetuned               | Fine-tuning de las últimas 25 capas       |
| Xception augmented               | Dataset aumentado con sin-cara sintéticas |
| Xception aug + finetuned         | Combinación de augmentation y fine-tuning |

### 4.5 Grad-CAM

Se implementó Gradient-weighted Class Activation Mapping (Grad-CAM) para visualizar las regiones de la imagen que más influyen en la predicción del modelo. El proceso consiste en:

1. Identificar la última capa convolucional del backbone (`block14_sepconv2_act` en Xception).
2. Calcular los gradientes de la clase predicha respecto a las activaciones de esa capa.
3. Ponderar los mapas de activación por el gradiente medio de cada canal.
4. Redimensionar el heatmap resultante y superponerlo sobre la imagen original.

Las visualizaciones Grad-CAM confirmaron que el modelo focaliza su atención en las regiones que el ojo humano también identifica como evocadoras de un rostro, validando cualitativamente el aprendizaje de la red.

---

## 5. Resultados y conclusiones destacables

### Resultados

La comparativa final entre los cuatro modelos entrenados sobre imágenes en escala de grises con Xception mostró que la combinación de augmentation y fine-tuning produjo el mejor rendimiento global, medido por AUC-ROC sobre el conjunto de test.

El análisis de la distribución de probabilidades predichas reveló que el modelo es altamente confiante en sus decisiones: la mayoría de las predicciones se concentran cerca de 0 o cerca de 1, con muy pocas predicciones en la zona de incertidumbre (0.3–0.7). Esto indica que el modelo aprendió representaciones discriminativas sólidas para la tarea.

### Conclusiones

**La pareidolia es detectable computacionalmente.** El modelo aprendió a reconocer los patrones estructurales que el cerebro humano asocia con rostros en objetos inanimados, lo que sugiere que estos patrones tienen correlatos visuales consistentes y aprendibles.

**Xception supera a EfficientNetB0 y ResNet50 en esta tarea.** Su arquitectura basada en convoluciones separables en profundidad es especialmente adecuada para capturar las texturas y formas locales que caracterizan las imágenes de pareidolia.

**La escala de grises no penaliza el rendimiento.** Eliminar la información de color y trabajar únicamente con luminancia no degradó significativamente la capacidad del modelo, lo que confirma que la tarea depende principalmente de la estructura espacial de las imágenes, no del color.

**El augmentation selectivo sobre la clase minoritaria es efectivo.** Aplicar transformaciones sintéticas únicamente sobre `sin-cara` ayudó a equilibrar el dataset sin introducir ruido artificial en la clase de interés.

**El orden en el fine-tuning es crítico.** Entrenar primero la cabeza con backbone congelado y aplicar fine-tuning en una segunda fase con learning rate bajo es esencial para preservar los pesos preentrenados y obtener mejoras reales de rendimiento.

**Grad-CAM valida cualitativamente el modelo.** Las regiones de alta activación coinciden con las zonas que el observador humano identifica como evocadoras de un rostro, lo que aporta interpretabilidad y confianza en las predicciones del modelo.
