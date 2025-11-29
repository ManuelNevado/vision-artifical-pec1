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
