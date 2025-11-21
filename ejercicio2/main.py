import cv2 as cv
import numpy as np
banner = """
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


def parte_a():
    print("Parte A")
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