import os

def listar_imagenes(carpeta):
    extensiones = ['.jpg', '.png', '.jpeg']
    return [f for f in os.listdir(carpeta) if os.path.splitext(f)[1].lower() in extensiones]
