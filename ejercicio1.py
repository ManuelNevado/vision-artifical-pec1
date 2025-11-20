import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
"""
seleccionar ROI, calcular el
histograma (de la imagen completa y de la ROI), aplicar operadores de transformación de la imagen
como umbralización con threshold manual, filtros de ruido, operaciones morfológicas, etc. Tenga
también en cuenta que la eliminación de algún tipo de ruido puede implicar la pérdida de las líneas
trazadas, por lo que deberá realizar una restauración aproximada (reconstruir una línea del mismo
grosor que el resto).
"""

def select_roi(path, x1, y1, x2, y2):
    """
    Selecciona y muestra una región de interés (ROI) de una imagen.
    
    Args:
        path (str): Ruta al archivo de imagen.
        x1 (int): Coordenada x del punto inicial de la ROI.
        y1 (int): Coordenada y del punto inicial de la ROI.
        x2 (int): Coordenada x del punto final de la ROI.
        y2 (int): Coordenada y del punto final de la ROI.
    
    Returns:
        numpy.ndarray: Imagen recortada correspondiente a la ROI.
    """
    image = cv.imread(path)
    cropped_image = image[y1:y2, x1:x2]
    cv.imshow("ROI", cropped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return cropped_image

def manual_threshold(path, threshold):
    # TODO implementar umbralización manual con un slider en la interfaz gráfica
    # TODO implementar distintos tipos de umbralización
    """
    Aplica umbralización binaria manual a una imagen.
    
    Args:
        path (str): Ruta al archivo de imagen.
        threshold (int): Valor del umbral (0-255) para la binarización.
    
    Returns:
        numpy.ndarray: Imagen umbralizada en formato binario.
    """
    image = cv.imread(path)
    _, thresholded_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold", thresholded_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return thresholded_image

def histograma(path):
    """
    Calcula y visualiza el histograma 2D (Hue vs Saturation) de una imagen en espacio HSV.
    
    Args:
        path (str): Ruta al archivo de imagen.
    
    Returns:
        numpy.ndarray: Histograma 2D de tamaño (180, 256) representando Hue y Saturation.
    """
    image = cv.imread(path)
    assert image is not None, "file could not be read, check with os.path.exists()"
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    histogram = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    plt.imshow(histogram,interpolation = 'nearest')
    plt.show()
    return histogram

def hist_plot(path):
    """
    Calcula el histograma de intensidades de píxeles de una imagen en escala de grises.
    
    Args:
        path (str): Ruta al archivo de imagen.
    
    Returns:
        tuple: Una tupla conteniendo:
            - count (list): Lista con la frecuencia de cada nivel de intensidad (0-255).
            - r (list): Lista con los valores de intensidad (0-255).
    """
    img = cv.imread(path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    count = []
    r= []
    for k in range(256):
        r.append(k)
        count1 = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] == k:
                    count1 += 1
        count.append(count1)
    
    return count, r







# Debuging, no leer para la correccion
if __name__ == "__main__":
    img_path = "/home/manuel/uned/vision_artificial/PEC1/data/DibujosNPT/N_303_AGL_TOTAL-ev1-r.png" 
    select_roi(img_path, 100, 100, 200, 200)
    manual_threshold(img_path, 127)
    histograma(img_path)