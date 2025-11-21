# Explicación de la Rotación de Imagen sin Recortes

Este documento detalla la lógica matemática y algorítmica detrás de la función `rotate_image_keep_whole` implementada en `main.py`.

## El Problema
Al rotar una imagen usando la transformación estándar (`cv.warpAffine` con una matriz de rotación básica), el tamaño del lienzo (canvas) se mantiene igual al de la imagen original. Esto provoca que las esquinas de la imagen rotada se "corten" o salgan del área visible.

## La Solución
Para evitar esto, necesitamos:
1.  Calcular el tamaño del nuevo rectángulo delimitador (bounding box) que contendrá la imagen rotada completa.
2.  Ajustar la matriz de rotación para que la imagen se centre en este nuevo lienzo más grande.

## Paso a Paso

### 1. Obtener Dimensiones y Centro
```python
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
```
Obtenemos el ancho y alto originales y calculamos el centro geométrico, que servirá como pivote para la rotación.

### 2. Matriz de Rotación Básica
```python
M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
```
Generamos la matriz de transformación afín para una rotación pura alrededor del centro.

### 3. Cálculo del Nuevo Tamaño (Trigonometría)
Para saber cuánto espacio ocupará la imagen rotada, proyectamos sus lados sobre los ejes X e Y usando el seno y coseno del ángulo de rotación.

```python
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])

nW = int((h * sin) + (w * cos))
nH = int((h * cos) + (w * sin))
```
*   **Nuevo Ancho (`nW`)**: Suma de la proyección de la altura y el ancho originales.
*   **Nueva Altura (`nH`)**: Suma de la proyección de la altura y el ancho originales.

### 4. Ajuste de Traslación (Centrado)
Esta es la parte crítica. La matriz original rota alrededor del centro antiguo `(cX, cY)`. Al cambiar el tamaño del lienzo a `(nW, nH)`, el centro cambia a `(nW/2, nH/2)`. Debemos sumar la diferencia a la componente de traslación de la matriz (la tercera columna).

```python
M[0, 2] += (nW / 2) - cX
M[1, 2] += (nH / 2) - cY
```

### 5. Aplicar la Transformación
Finalmente, aplicamos `warpAffine` especificando el **nuevo tamaño** `(nW, nH)`. OpenCV crea un lienzo de ese tamaño y mapea los píxeles usando la matriz ajustada.

```python
return cv.warpAffine(image, M, (nW, nH))
```
