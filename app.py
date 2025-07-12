
import gradio as gr
import cv2
import numpy as np
import os

RUTA_IMAGENES = "./img"

def cargar_imagen(ruta):
    return cv2.imread(ruta)

def convertir_espacio_color(imagen, modo):
    if modo == "RGB":
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    elif modo == "HSV":
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    elif modo == "LAB":
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2Lab)
    elif modo == "GRAYSCALE":
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen

def ajustar_brillo_contraste(imagen, brillo=0, contraste=0):
    return cv2.convertScaleAbs(imagen, alpha=1 + contraste / 100, beta=brillo)

def aplicar_filtro(imagen, tipo, intensidad=5):
    ksize = max(1, intensidad * 2 + 1)
    if tipo == "Media":
        return cv2.blur(imagen, (ksize, ksize))
    elif tipo == "Gaussiano":
        return cv2.GaussianBlur(imagen, (ksize, ksize), 0)
    elif tipo == "Mediana":
        return cv2.medianBlur(imagen, ksize)
    elif tipo == "Bilateral":
        return cv2.bilateralFilter(imagen, 9, 75, 75)
    return imagen


def umbralizar(imagen, tipo="Binario", valor=127):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if tipo == "Binario":
        _, binaria = cv2.threshold(gris, valor, 255, cv2.THRESH_BINARY)
    elif tipo == "Otsu":
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif tipo == "Adaptativo":
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    else:
        binaria = gris
    return binaria

    if tipo == "Binario":
        _, binaria = cv2.threshold(gris, valor, 255, cv2.THRESH_BINARY)
    elif tipo == "Otsu":
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif tipo == "Adaptativo":
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    return binaria

def asegurar_rgb(imagen):
    if imagen is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if len(imagen.shape) == 2:
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    elif imagen.shape[2] == 4:
        return cv2.cvtColor(imagen, cv2.COLOR_BGRA2RGB)
    elif imagen.shape[2] == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    return imagen

def listar_imagenes(carpeta):
    extensiones = ['.jpg', '.png', '.jpeg']
    archivos = [f for f in os.listdir(carpeta) if os.path.splitext(f)[1].lower() in extensiones]
    print("IMÁGENES DETECTADAS:", archivos)
    return archivos

def procesar(imagen_nombre, espacio_color, brillo, contraste, filtro, intensidad, umbral_tipo, umbral_valor):
    ruta = os.path.join(RUTA_IMAGENES, imagen_nombre)
    img_original = cargar_imagen(ruta)
    if img_original is None:
        return [np.zeros((100,100,3), dtype=np.uint8)] * 5
    img_color = convertir_espacio_color(img_original, espacio_color)
    img_ajustada = ajustar_brillo_contraste(img_color, brillo, contraste)
    img_filtrada = aplicar_filtro(img_ajustada, filtro, intensidad)
    img_umbral = umbralizar(img_filtrada, umbral_tipo, umbral_valor)
    return [
        asegurar_rgb(img_original),
        asegurar_rgb(img_color),
        asegurar_rgb(img_ajustada),
        asegurar_rgb(img_filtrada),
        asegurar_rgb(img_umbral)
    ]

interfaz = gr.Interface(
    fn=procesar,
    inputs=[
        gr.Dropdown(choices=listar_imagenes(RUTA_IMAGENES), label="Imagen"),
        gr.Radio(["RGB", "HSV", "LAB", "GRAYSCALE"], label="Espacio de color"),
        gr.Slider(-100, 100, label="Brillo"),
        gr.Slider(-100, 100, label="Contraste"),
        gr.Radio(["Media", "Gaussiano", "Mediana", "Bilateral"], label="Filtro"),
        gr.Slider(1, 10, step=1, label="Intensidad Filtro"),
        gr.Radio(["Binario", "Adaptativo", "Otsu"], label="Tipo de Umbral"),
        gr.Slider(0, 255, step=1, label="Valor Umbral")
    ],
    outputs=[
        gr.Image(label="Original"),
        gr.Image(label="Convertida"),
        gr.Image(label="Brillo/Contraste"),
        gr.Image(label="Filtrada"),
        gr.Image(label="Umbralizada")
    ],
    title="Procesador de Imágenes con OpenCV",
    description="Selecciona una imagen y ajusta sus propiedades para ver el resultado."
)

interfaz.launch(inline=False, share=False, show_error=True, debug=True)
