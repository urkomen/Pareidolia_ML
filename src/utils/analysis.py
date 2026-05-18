import os, sys
import random
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image

# Añadir src a path para importar utils
# sys.path.append(os.path.join(os.getcwd(), '../../src'))
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(PROJECT_ROOT)
from src.utils import IMG_SIZE


def renombrar_imagenes_cara(carpeta):
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}

    archivos = [
        f for f in os.listdir(carpeta)
        if os.path.isfile(os.path.join(carpeta, f))
        and os.path.splitext(f)[1].lower() in extensiones_validas
    ]

    archivos.sort()

    # Comprobación previa: detectar si ya existen archivos con nomenclatura 'cara_XXX'
    ya_renombrados = [f for f in archivos if f.lower().startswith('cara_') and f[4:7].isdigit()]

    if ya_renombrados:
        print(f'⚠️  Se encontraron {len(ya_renombrados)} archivo(s) con nomenclatura "cara_XXX":')
        for f in ya_renombrados:
            print(f'   - {f}')
        respuesta = input('/n¿Deseas continuar de todas formas? (s/n): ').strip().lower()
        if respuesta != 's':
            print('Operación cancelada.')
            return

    for i, archivo in enumerate(archivos, start=1):
        extension = os.path.splitext(archivo)[1].lower()
        
        # Nombre base inicial
        base_num = i
        nuevo_nombre = f'cara_{base_num:03d}{extension}'
        destino = os.path.join(carpeta, nuevo_nombre)

        # Si ya existe, buscar el siguiente disponible
        while os.path.exists(destino):
            base_num += 1
            nuevo_nombre = f'cara_{base_num:03d}{extension}'
            destino = os.path.join(carpeta, nuevo_nombre)

        origen = os.path.join(carpeta, archivo)
        os.rename(origen, destino)
        print(f'{archivo} → {nuevo_nombre}')
    
def renombrar_imagenes_sincara(carpeta):
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}

    archivos = [
        f for f in os.listdir(carpeta)
        if os.path.isfile(os.path.join(carpeta, f))
        and os.path.splitext(f)[1].lower() in extensiones_validas
    ]

    archivos.sort()

    # Comprobación previa: detectar si ya existen archivos con nomenclatura 'sin-cara_XXX'
    ya_renombrados = [f for f in archivos if f.lower().startswith('sin-cara_') and f[8:11].isdigit()]

    if ya_renombrados:
        print(f'⚠️  Se encontraron {len(ya_renombrados)} archivo(s) con nomenclatura "sin-cara_XXX":')
        for f in ya_renombrados:
            print(f'   - {f}')
        respuesta = input('/n¿Deseas continuar de todas formas? (s/n): ').strip().lower()
        if respuesta != 's':
            print('Operación cancelada.')
            return

    for i, archivo in enumerate(archivos, start=1):
        extension = os.path.splitext(archivo)[1].lower()
        
        # Nombre base inicial
        base_num = i
        nuevo_nombre = f'sin-cara_{base_num:03d}{extension}'
        destino = os.path.join(carpeta, nuevo_nombre)

        # Si ya existe, buscar el siguiente disponible
        while os.path.exists(destino):
            base_num += 1
            nuevo_nombre = f'sin-cara_{base_num:03d}{extension}'
            destino = os.path.join(carpeta, nuevo_nombre)

        origen = os.path.join(carpeta, archivo)
        os.rename(origen, destino)
        print(f'{archivo} → {nuevo_nombre}')


def renombrar_predicciones(carpeta):
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}

    archivos = [
        f for f in os.listdir(carpeta)
        if os.path.isfile(os.path.join(carpeta, f))
        and os.path.splitext(f)[1].lower() in extensiones_validas
    ]

    archivos.sort()

    # Comprobación previa: detectar si ya existen archivos con nomenclatura 'predict_XXX'
    ya_renombrados = [f for f in archivos if f.lower().startswith('predict_') and f[8:11].isdigit()]

    if ya_renombrados:
        print(f'⚠️  Se encontraron {len(ya_renombrados)} archivo(s) con nomenclatura "predict_XXX":')
        for f in ya_renombrados:
            print(f'   - {f}')
        respuesta = input('/n¿Deseas continuar de todas formas? (s/n): ').strip().lower()
        if respuesta != 's':
            print('Operación cancelada.')
            return

    for i, archivo in enumerate(archivos, start=1):
        extension = os.path.splitext(archivo)[1].lower()
        
        # Nombre base inicial
        base_num = i
        nuevo_nombre = f'predict_{base_num:03d}{extension}'
        destino = os.path.join(carpeta, nuevo_nombre)

        # Si ya existe, buscar el siguiente disponible
        while os.path.exists(destino):
            base_num += 1
            nuevo_nombre = f'predict_{base_num:03d}{extension}'
            destino = os.path.join(carpeta, nuevo_nombre)

        origen = os.path.join(carpeta, archivo)
        os.rename(origen, destino)
        print(f'{archivo} → {nuevo_nombre}')
        

# Funciones utilitarias para EDA
def list_images(folder):
    return sorted(glob(os.path.join(folder, '*', '*')))

def sample_images(folder, n=20):
    imgs = list_images(folder)
    return random.sample(imgs, min(n, len(imgs)))

def load_pil(path):
    return Image.open(path).convert('RGB')


def analyze_sizes(paths):
    sizes = []
    for p in paths:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)  # (width, height)
        except:
            continue
    sizes = np.array(sizes)
    return sizes


def mean_color(path):
    im = load_pil(path)
    arr = np.array(im).astype(np.float32)/255.0
    return arr.mean(axis=(0,1))


def show_random_by_class(folder, cls, n=9):
    paths = glob(os.path.join(folder, cls, '*'))
    paths = random.sample(paths, min(n, len(paths)))
    plt.figure(figsize=(8,8))
    for i,p in enumerate(paths):
        plt.subplot(3,3,i+1)
        plt.imshow(load_pil(p).resize((224,224)))
        plt.axis('off')
    plt.suptitle(f'{folder} / {cls} (muestra)')
    plt.show()


def symmetry_score(path):
    im = load_pil(path).resize(IMG_SIZE)
    arr = np.array(im).astype(np.float32)/255.0
    # Imágenes en escala de grises (2D) → añadir dimensión canal
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    left = arr[:, :arr.shape[1]//2, :].mean(axis=2)
    right = arr[:, arr.shape[1] - arr.shape[1]//2:, :].mean(axis=2)[:, ::-1]
    # recortar a la misma forma si hay diferencia
    min_cols = min(left.shape[1], right.shape[1])
    left = left[:, :min_cols]
    right = right[:, :min_cols]
    l, r = left.flatten(), right.flatten()
    # Protección: si alguno es constante, la correlación no tiene sentido
    if l.std() < 1e-6 or r.std() < 1e-6:
        return 0.0  # o np.nan si prefieres excluirlas del análisis
    return np.corrcoef(l, r)[0, 1]


def find_outliers(paths):
    bad = []
    for p in paths:
        try:
            im = Image.open(p)
            w,h = im.size
            if w < 50 or h < 50:
                bad.append((p, 'too_small', (w,h)))
        except Exception as e:
            bad.append((p, 'error', str(e)))
    return bad