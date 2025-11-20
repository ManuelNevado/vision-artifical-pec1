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
    # Seleccionar ROI en una imagen dada unas coordenadas
    image = cv.imread(path)
    cropped_image = image[y1:y2, x1:x2]
    cv.imshow("ROI", cropped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return cropped_image

def manual_threshold(path, threshold):
    # Aplicar umbralización manual
    image = cv.imread(path)
    _, thresholded_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold", thresholded_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return thresholded_image

def histograma(path):
    # Calcular histograma de una imagen
    image = cv.imread(path)
    assert image is not None, "file could not be read, check with os.path.exists()"
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    histogram = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    plt.imshow(histogram,interpolation = 'nearest')
    plt.show()
    return histogram

def hist_plot(path):
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
    
    plt.stem(r, count)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.title("Histograma")
    plt.show()

if __name__ == "__main__":
    img_path = "/home/manuel/uned/vision_artificial/PEC1/data/DibujosNPT/N_303_AGL_TOTAL-ev1-r.png" 
    select_roi(img_path, 100, 100, 200, 200)
    manual_threshold(img_path, 127)
    histograma(img_path)