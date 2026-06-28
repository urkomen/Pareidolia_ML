'''
Página de predicción — Pareidolia ML
'''

import sys
import numpy as np
import cv2 as cv
import streamlit as st
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.prediction import compute_gradcam, overlay_heatmap

MODEL_DIR = Path(__file__).parent.parent.parent / 'model'
TARGET_SIZE = (299, 299)

MODEL_OPTIONS = {
    'Xception (Augmented + Fine-tuned) — Recomendado': MODEL_DIR / 'production' / 'Xception_augmented_finetuned.keras',
    'Xception (Augmented)': MODEL_DIR / 'Xception_augmented.keras',
    'Xception (Fine-tuned)': MODEL_DIR / 'Xception_finetuned.keras',
    'Xception (Original)': MODEL_DIR / 'Xception.keras',
}


@st.cache_resource
def cargar_modelo(model_path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def preprocess(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert('RGB').resize(TARGET_SIZE)) / 255.0
    return np.expand_dims(arr, axis=0)


def predecir(model, image_array: np.ndarray, threshold: float):
    prob = float(model.predict(image_array, verbose=0)[0][0])
    es_cara = prob >= threshold
    return 'CARA' if es_cara else 'SIN CARA', prob, prob if es_cara else 1 - prob


# ── App ──────────────────────────────────────

st.set_page_config(page_title='Predicción — Pareidolia ML', page_icon='🔍', layout='wide')
st.title('🔍 Predicción')
st.page_link('streamlit.py', label='← Volver a Inicio')

# Sidebar
st.sidebar.header('Configuración')
model_name = st.sidebar.selectbox('Modelo:', list(MODEL_OPTIONS.keys()))
threshold = st.sidebar.slider('Umbral de clasificación', 0.0, 1.0, 0.5, 0.05)
mostrar_gradcam = st.sidebar.checkbox('Mostrar Grad-CAM', value=True)

# Carga de imagen
st.subheader('Cargar imagen')
col_up, col_cam = st.columns([2, 1])
with col_up:
    uploaded_file = st.file_uploader('JPG, JPEG o PNG', type=['jpg', 'jpeg', 'png'])
with col_cam:
    usar_camara = st.checkbox('Usar cámara web')
    if usar_camara:
        foto = st.camera_input('Toma una foto')
        if foto:
            uploaded_file = foto

if not uploaded_file:
    st.info('Carga una imagen para comenzar.')
    st.stop()

image = Image.open(uploaded_file)

col_img, col_res = st.columns(2)
with col_img:
    st.image(image, caption=f'{image.size[0]}×{image.size[1]} px', use_container_width=True)

with col_res:
    if st.button('Ejecutar predicción', type='primary', use_container_width=True):
        model = cargar_modelo(str(MODEL_OPTIONS[model_name]))
        image_array = preprocess(image)
        label, prob, confianza = predecir(model, image_array, threshold)

        color = 'green' if label == 'CARA' else 'red'
        st.markdown(f'<h2 style="color:{color}">{label}</h2>', unsafe_allow_html=True)
        st.metric('Probabilidad (cara)', f'{prob:.2%}')
        st.metric('Confianza', f'{confianza:.2%}')

        if confianza < 0.6:
            st.warning('Predicción poco segura — imagen ambigua.')

        if mostrar_gradcam:
            st.markdown('---')
            st.caption('Grad-CAM — zona de atención del modelo')
            with st.spinner('Generando Grad-CAM...'):
                try:
                    heatmap = compute_gradcam(model, image_array, 'block14_sepconv2_act')
                    img_rgb = np.array(image.convert('RGB').resize(TARGET_SIZE))
                    result = overlay_heatmap(heatmap, img_rgb)
                    st.image(cv.cvtColor(result, cv.COLOR_BGR2RGB), use_container_width=True)
                except Exception as e:
                    st.error(f'Error en Grad-CAM: {e}')

        with st.expander('Detalles técnicos'):
            st.write(f'**Modelo:** {model_name}')
            st.write(f'**Umbral:** {threshold}')
            st.write(f'**Probabilidad bruta:** {prob:.6f}')
