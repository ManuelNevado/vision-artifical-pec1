import cv2 as cv
import numpy as np
import os
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
    os.chdir("data")
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


def parte_b():
    print("Parte B")


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