'''
Home — Pareidolia ML
Ejecutar desde la raíz del proyecto: streamlit run src/utils/streamlit.py
'''

import streamlit as st

st.set_page_config(page_title='Pareidolia', page_icon='👁️', layout='centered')

st.title('👁️👄👁️ Pareidolia - Detector de caras')
st.subheader('¿Ves una cara donde no la hay?')

st.markdown('---')

st.markdown('''
## ¿Qué es la pareidolia?

La **pareidolia** es el fenómeno psicológico por el que el cerebro humano percibe patrones
familiares —especialmente caras— en estímulos aleatorios o ambiguos: nubes, manchas,
superficies de madera, tostadas, sombras...

No es una ilusión óptica al uso. Es una respuesta automática e involuntaria del sistema
visual: nuestro cerebro está tan especializado en detectar caras que a veces "ve" una
donde no existe ninguna.
''')

st.markdown('---')

st.markdown('''
## Cómo funciona este detector

Este proyecto entrena una red neuronal convolucional (**Xception**) para replicar
ese comportamiento humano: dado una imagen, el modelo predice si contiene o no
una cara percibida por pareidolia.

El modelo ha sido entrenado con un dataset de imágenes etiquetadas manualmente
en dos clases:
- **cara** — imágenes donde se percibe una cara (416 ejemplos)
- **sin-cara** — imágenes sin percepción facial (652 ejemplos)

Además, la app incluye **Grad-CAM**: una técnica de visualización que muestra
qué zona de la imagen ha activado la respuesta del modelo, permitiendo entender
*dónde* ha "visto" la cara.
''')

st.markdown('---')

st.markdown('''
## Cómo usar la app

1. Haz clic en el botón de abajo para ir a la página de predicción
2. Carga una imagen (JPG o PNG) o usa la cámara web
3. Selecciona el modelo en la barra lateral *(recomendado: Augmented + Fine-tuned)*
4. Pulsa **Ejecutar predicción**
5. Observa el resultado y el mapa Grad-CAM
''')

st.markdown('<br>', unsafe_allow_html=True)
st.page_link('pages/1_Prediccion.py', label='Ir al detector →', icon='🔍')
