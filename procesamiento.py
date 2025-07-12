import cv2
import numpy as np

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
    return binaria
