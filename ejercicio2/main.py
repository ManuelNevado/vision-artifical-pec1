import cv2 as cv
import numpy as np
import os
import math


banner = r"""
  _______                   __                                _                         _ _                  _           
 |__   __|                 / _|                              (_)                       | (_)                | |          
    | |_ __ __ _ _ __  ___| |_ ___  _ __ _ __ ___   __ _  ___ _  ___  _ __   ___  ___  | |_ _ __   ___  __ _| | ___  ___ 
    | | '__/ _` | '_ \/ __|  _/ _ \| '__| '_ ` _ \ / _` |/ __| |/ _ \| '_ \ / _ \/ __| | | | '_ \ / _ \/ _` | |/ _ \/ __|
    | | | | (_| | | | \__ \ || (_) | |  | | | | | | (_| | (__| | (_) | | | |  __/\__ \ | | | | | |  __/ (_| | |  __/\__ \
    |_|_|  \__,_|_| |_|___/_| \___/|_|  |_| |_| |_|\__,_|\___|_|\___/|_| |_|\___||___/ |_|_|_| |_|\___|\__,_|_|\___||___/
  _____  ______ _____ __                                                                                                 
 |  __ \|  ____/ ____/_ |                                                                                                
 | |__) | |__ | |     | |                                                                                                
 |  ___/|  __|| |     | |                                                                                                
 | |    | |___| |____ | |                                                                                                
 |_|    |______\_____||_|                                                                                                
                                                                                                                         
                                                                                                                         
"""



def rotate_image_keep_whole(image, angle):
    """
    Rotates an image by a given angle while keeping the whole image visible.
    """
    # Get image dimensions
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))


def parte_a():
    """
    Reescalar las imágenes a 200x200
    Rotarlas 45 grados tomando como centro de rotacion el centro de la imagen
    configurar el tamaño de la imagen para que se muestre toda la imagen
    """
    img1 = cv.imread("zigzag.jpg")
    img2 = cv.imread("brainLabels.png")

    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return

    img1 = cv.resize(img1, (200, 200))
    img2 = cv.resize(img2, (200, 200))

    # Rotacion de 45 grados manteniendo toda la imagen
    img1_rotated = rotate_image_keep_whole(img1, 45)
    img2_rotated = rotate_image_keep_whole(img2, 45)

    cv.imshow("img1 Original", img1)
    cv.imshow("img1 Rotated", img1_rotated)
    cv.imshow("img2 Original", img2)
    cv.imshow("img2 Rotated", img2_rotated)
    
    print("Press any key to close windows...")
    cv.waitKey(0)
    cv.destroyAllWindows()


def parte_b1():
    """
    T1: Reescalar a un cuarto en ambos ejes la imagen
    T2: Inclinar (shear) -30 grados la imagen obtenida de T1
    T3: girar 90 grados a la izquierda la imagen obtenida de T2

    Se muestra la imagen despues de cada transformacion
    """
    img1 = cv.imread("zigzag.jpg")
    if img1 is None:
        print("Error: Image not found")
        return

    h, w = img1.shape[:2]
    
    # --- T1: Reescalado ---
    M_resize = np.float32([
        [1/4, 0, 0],
        [0, 1/4, 0]
    ])
    img1 = cv.warpAffine(img1, M_resize, (w//4, h//4))
    cv.imshow("img1 T1", img1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # --- T2: Shear -30 grados ---
    rows, cols = img1.shape[:2]
    
    angle_degrees = -30
    angle_radians = np.radians(angle_degrees)
    shx = np.tan(angle_radians)

    # Calcular el desplazamiento necesario (porque shx es negativo, la imagen se va a la izquierda)
    abs_shift = abs(rows * shx)
    
    # Crear la matriz de shear con la traslación incluida
    M_shear = np.float32([
        [1, shx, abs_shift], 
        [0, 1, 0]
    ])

    # Calcular nuevo ancho
    new_width = cols + int(abs_shift)
    
    # APLICAR UNA SOLA VEZ
    img_sheared = cv.warpAffine(img1, M_shear, (new_width, rows))

    cv.imshow("img1 T2", img_sheared)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # --- T3: Rotación 90 grados ---
    h_s, w_s = img_sheared.shape[:2]
    (cX, cY) = (w_s // 2, h_s // 2)
    
    M_rot = cv.getRotationMatrix2D((cX, cY), 90, 1)
    
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])

    nW = int((h_s * sin) + (w_s * cos))
    nH = int((h_s * cos) + (w_s * sin))
    
    M_rot[0, 2] += (nW / 2) - cX
    M_rot[1, 2] += (nH / 2) - cY
    
    img_final = cv.warpAffine(img_sheared, M_rot, (nW, nH))
    
    cv.imshow("img1 T3", img_final)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def parte_b2():
    # Se aplican todas las transformaciones de B1.1 de una sola vez conmutando matrices de transformacion
    img1 = cv.imread("zigzag.jpg")
    if img1 is None:
        print("Error: Image not found")
        return
    
    # Para multiplicar matrices de transformacion afin, necesitamos usar coordenadas homogeneas (3x3)
    # Añadimos la fila [0, 0, 1] a cada matriz
    
    M_resize_3x3 = np.eye(3)
    M_resize_3x3[:2] = np.float32([
        [1/4, 0, 0],
        [0, 1/4, 0]
    ])
    
    M_shear_3x3 = np.eye(3)
    M_shear_3x3[:2] = np.float32([
        [1, np.tan(np.radians(-30)), 0],
        [0, 1, 0]
    ])
    
    M_rot_3x3 = np.eye(3)
    M_rot_3x3[:2] = np.float32([
        [math.cos(math.radians(90)), math.sin(math.radians(90)), 0],
        [-math.sin(math.radians(90)), math.cos(math.radians(90)), 0]
    ])
    
    # Multiplicacion de matrices (orden inverso: T3 * T2 * T1)
    M_T_3x3 = np.dot(M_rot_3x3, np.dot(M_shear_3x3, M_resize_3x3))
    
    # --- Calculo del Bounding Box y Ajuste de Traslacion ---
    h, w = img1.shape[:2]
    
    # Definir las 4 esquinas de la imagen original (coordenadas homogeneas)
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T # Transponer para que sean vectores columna (3x4)
    
    # Aplicar la transformacion a las esquinas
    transformed_corners = np.dot(M_T_3x3, corners)
    
    # Normalizar (aunque la ultima fila deberia ser 1, es buena practica)
    transformed_corners = transformed_corners[:2, :]
    
    # Encontrar los limites (min/max x, min/max y)
    min_x = transformed_corners[0, :].min()
    max_x = transformed_corners[0, :].max()
    min_y = transformed_corners[1, :].min()
    max_y = transformed_corners[1, :].max()
    
    # Calcular el desplazamiento necesario para que empiece en (0,0)
    translation_x = -min_x
    translation_y = -min_y
    
    # Añadir la traslacion a la matriz combinada
    M_T_3x3[0, 2] += translation_x
    M_T_3x3[1, 2] += translation_y
    
    # Calcular el nuevo tamaño del canvas
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    
    # Extraemos la matriz 2x3 resultante para warpAffine
    M_T = M_T_3x3[:2]
    
    img_final = cv.warpAffine(img1, M_T, (new_w, new_h))
    
    cv.imshow("img1 Tc", img_final)
    cv.waitKey(0)
    cv.destroyAllWindows()
        
    

def parte_b():
    #Menu parte B
    print("Parte B")
    print("1. B1.1")
    print("2. B1.2")
    print("3. Salir")
    option = int(input("Introduce una opcion: "))
    while option != 3:
        if option == 1:
            parte_b1()
        elif option == 2:
            parte_b2()
        option = int(input("Introduce una opcion: "))

def parte_c():
    print("Parte C")


def menu():
    print("""
    1. Parte A. Reescalar y rotar las imagenes manteniendo toda la informacion
    2. Parte B. 
    3. Parte C. 
    4. Salir
    """)
    option = int(input("Introduce una opcion: "))
    return option

def main():
    os.chdir("data")
    print(banner)
    mode = menu()
    while mode != 4:
        if mode == 1:
            parte_a()
        elif mode == 2:
            parte_b()
        elif mode == 3:
            parte_c()
        mode = menu()   
    print("Adios")

if __name__ == "__main__":
    main()