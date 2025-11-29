# Instrucciones de Ejecución - Ejercicio 2

Este documento describe cómo ejecutar la aplicación principal del Ejercicio 2.

## Prerrequisitos

*   Python 3 instalado.
*   Librerías listadas en `requirements.txt`.

## Instalación de Dependencias

Antes de ejecutar la aplicación, asegúrate de instalar las dependencias necesarias. Abre una terminal en la raíz del proyecto (`PEC1`) y ejecuta:

```bash
pip install -r requirements.txt
```

## Ejecución de la Aplicación

La aplicación debe ejecutarse desde la **raíz del proyecto** (`PEC1`) para que pueda localizar correctamente el directorio de datos (`data`).

1.  Abre una terminal.
2.  Navega hasta el directorio `PEC1`.
3.  Ejecuta el siguiente comando:

```bash
python ejercicio2/main.py
```

## Notas Importantes

*   **Directorio de Trabajo**: El script `main.py` cambia internamente el directorio de trabajo a `data` (línea `os.chdir("data")`). Por esta razón, es crucial lanzar el script desde una ubicación donde la carpeta `data` sea accesible o relativa a la estructura esperada. Dado que la carpeta `data` se encuentra en la raíz de `PEC1`, ejecutar el script desde `PEC1` asegura que el programa funcione correctamente.
*   **Imágenes**: El programa espera encontrar las imágenes `zigzag.jpg`, `brainLabels.png` y `tableroAjedrez.png` dentro de la carpeta `data`.
