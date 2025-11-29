# Resumen y Análisis de Algoritmos de Flujo Óptico

Este documento resume el análisis de los scripts de flujo óptico ubicados en `data/opticalFlow`. Aunque en la solicitud se indicó dos veces la ruta de `lukas_kanade_track.py`, se ha incluido también el análisis de `optical_flow1.py` (presente en el mismo directorio) para ofrecer una comparativa útil de las dos técnicas principales implementadas.

## 1. Análisis de `lukas_kanade_track.py`

Este script implementa una técnica de **Flujo Óptico Esparso (Sparse Optical Flow)**.

*   **Algoritmo Principal**: Utiliza el método de **Lucas-Kanade** con pirámides (`cv.calcOpticalFlowPyrLK`). Este método asume que el flujo es constante en una pequeña vecindad local alrededor del píxel de interés.
*   **Selección de Características**: No calcula el flujo para toda la imagen, sino solo para puntos de interés "buenos" (esquinas fuertes). Utiliza `cv.goodFeaturesToTrack` (detector Shi-Tomasi) para inicializar los puntos a seguir.
*   **Mecanismo de Seguimiento**:
    *   Mantiene una lista de "tracks" (trayectorias).
    *   En cada frame, calcula la nueva posición de los puntos antiguos.
    *   **Verificación (Back-tracking)**: Implementa una técnica robusta de validación. Calcula el flujo hacia adelante ($t \to t+1$) y luego hacia atrás ($t+1 \to t$). Si el punto calculado al volver atrás difiere del original en más de 1 píxel (`d < 1`), el punto se descarta. Esto elimina errores de seguimiento.
    *   **Reinicialización**: Cada 5 frames (`detect_interval`), busca nuevas características para añadir a la lista de seguimiento, asegurando que no se pierdan todos los puntos con el tiempo.
*   **Visualización**: Dibuja las trayectorias (colas) de los puntos seguidos en verde.

## 2. Análisis de `optical_flow1.py`

Este script implementa una técnica de **Flujo Óptico Denso (Dense Optical Flow)**.

*   **Algoritmo Principal**: Utiliza el algoritmo de **Gunnar Farneback** (`cv.calcOpticalFlowFarneback`). Este método calcula el desplazamiento para **todos** los píxeles de la imagen basándose en la expansión polinomial.
*   **Selección de Características**: No hay selección; se procesa la imagen completa.
*   **Visualización**: Ofrece varias formas de visualizar el campo de movimiento denso:
    *   **Vectores (Grid)**: Dibuja una rejilla de líneas verdes que indican la dirección y magnitud del movimiento en cada zona.
    *   **Mapa de Color HSV**: Mapea la dirección del movimiento al Matiz (Hue) y la magnitud al Valor/Saturación, creando una imagen coloreada continua.
    *   **Glitch/Warp**: Deforma la imagen anterior basándose en el flujo calculado.

## 3. Comparativa: Similitudes y Diferencias

### Similitudes
*   **Base Tecnológica**: Ambos utilizan la librería OpenCV (`cv2`) y NumPy.
*   **Entrada**: Ambos procesan vídeo frame a frame (ya sea de archivo o webcam).
*   **Objetivo**: Ambos buscan estimar el movimiento aparente de los objetos en la escena.

### Diferencias

| Característica | `lukas_kanade_track.py` (Esparso) | `optical_flow1.py` (Denso) |
| :--- | :--- | :--- |
| **Cobertura** | Solo puntos de interés (esquinas). | Todos los píxeles de la imagen. |
| **Algoritmo** | Lucas-Kanade (local, diferencial). | Farneback (global, expansión polinomial). |
| **Coste Computacional** | Generalmente menor (depende del nº de puntos). | Mayor (procesa toda la imagen). |
| **Robustez** | Alta para puntos específicos (gracias al back-tracking). | Proporciona una visión global pero puede tener ruido. |
| **Información** | Trayectorias históricas (historia del movimiento). | Campo de velocidad instantáneo (movimiento actual). |
| **Uso Ideal** | Seguimiento de objetos (tracking), SLAM. | Segmentación de movimiento, estructura desde movimiento. |

## 4. Limitaciones de Lucas-Kanade en Regiones Suavizadas

Una observación importante sobre la implementación en `lukas_kanade_track.py` es que **no detecta movimiento en regiones suavizadas o de color uniforme**. Esto se debe a dos razones fundamentales:

1.  **Problema de la Apertura y Gradientes**: El algoritmo de Lucas-Kanade se basa en resolver un sistema de ecuaciones que depende de los gradientes espaciales de la imagen ($I_x, I_y$). En regiones planas o suaves, estos gradientes son cercanos a cero, lo que hace que el sistema sea matemáticamente irresoluble (matriz singular) o muy inestable. Sin bordes o texturas, es imposible determinar localmente si un píxel se ha movido.

2.  **Selección Explícita de Características**: Para evitar cálculos erróneos en estas zonas problemáticas, el script utiliza `cv.goodFeaturesToTrack` (detector Shi-Tomasi). Este detector filtra explícitamente las regiones donde los valores propios del tensor de estructura son pequeños (es decir, regiones planas), seleccionando únicamente **esquinas fuertes** donde el movimiento puede ser rastreado con fiabilidad.

## 5. Métodos de Visualización del Flujo Óptico

En el contexto de los scripts analizados, se pueden identificar tres formas principales de representar visualmente la información del flujo óptico:

1.  **Campo Vectorial (Grid de Vectores)**:
    *   **Descripción**: Se dibuja una rejilla de líneas o flechas sobre la imagen original. Cada línea nace en un punto $ y termina en $.
    *   **Interpretación**: La longitud de la línea indica la **velocidad** (magnitud del desplazamiento) y la orientación indica la **dirección** del movimiento.
    *   **Implementación**: Ver función `draw_flow` en `optical_flow1.py`. Es útil para ver el flujo detallado en zonas específicas, aunque puede saturar la imagen si se dibujan todos los vectores (por eso se usa un `step` o paso).

2.  **Mapa de Color (Codificación HSV)**:
    *   **Descripción**: Se genera una imagen sintética donde el movimiento de cada píxel se codifica mediante color.
    *   **Interpretación**:
        *   **Matiz (Hue)**: Representa la **dirección** del movimiento (ej. rojo=derecha, azul=arriba).
        *   **Valor/Saturación**: Representa la **magnitud** del movimiento (más brillante/intenso = mayor velocidad).
    *   **Implementación**: Ver función `draw_hsv` en `optical_flow1.py`. Es excelente para visualizar el movimiento denso de forma global y continua, permitiendo identificar objetos en movimiento rápidamente.

3.  **Trayectorias (Tracking)**:
    *   **Descripción**: En lugar de mostrar el movimiento instantáneo, se dibuja el camino recorrido por puntos específicos a lo largo del tiempo.
    *   **Interpretación**: Muestra la **historia** del movimiento. Las líneas conectan las posiciones pasadas con la actual.
    *   **Implementación**: Ver clase `App` en `lukas_kanade_track.py`. Es ideal para el flujo óptico esparso, ya que permite visualizar el comportamiento dinámico de objetos concretos a través de múltiples frames.
