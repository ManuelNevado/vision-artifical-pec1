# Memoria Justificativa - Ejercicio 2

## Introducción
Este documento detalla las decisiones de diseño e implementación adoptadas para la resolución de los distintos apartados del Ejercicio 2. El objetivo principal ha sido asegurar la corrección matemática de las transformaciones geométricas y la preservación de la información visual.

## Parte A: Rotación y Preservación de Información
La implementación estándar de la rotación mediante matrices afines mantiene por defecto el tamaño del lienzo original. Esto conlleva un problema inevitable: al girar una imagen rectangular, las esquinas se proyectan fuera de los límites originales, resultando en una pérdida de información visual (recorte).

Para solucionar este problema, se ha decidido implementar una función personalizada que recalcula las dimensiones del lienzo. La decisión se basa en la trigonometría: proyectamos la altura y anchura originales sobre los nuevos ejes rotados para obtener el "bounding box" mínimo necesario que contenga toda la imagen.

Además del redimensionamiento, es imperativo ajustar la componente de traslación de la matriz de rotación. Si solo ampliáramos el lienzo, la imagen rotada seguiría pivotando sobre el centro antiguo, quedando desplazada o parcialmente fuera de la nueva vista. Por ello, se calcula el nuevo centro geométrico y se añade un desplazamiento (offset) a la matriz afín para garantizar que la imagen rotada quede perfectamente centrada en el nuevo lienzo.

## Parte B1: Transformaciones Secuenciales y Composición
El ejercicio requería abordar la transformación compuesta (escalado, cizallamiento y rotación) desde dos perspectivas complementarias.

En primer lugar, se ha optado por una aplicación secuencial de las transformaciones. Esta decisión responde a una necesidad de verificación y depuración paso a paso. Al visualizar el resultado intermedio, podemos confirmar que cada operación individual produce el efecto geométrico esperado. Para el cizallamiento (shear), fue necesario implementar un ajuste manual del ancho del lienzo y una traslación compensatoria para evitar que los píxeles desplazados hacia la izquierda (por el ángulo negativo) se perdieran fuera de los límites de la imagen.

En segundo lugar, se justifica el uso de la composición de matrices mediante coordenadas homogéneas (matrices de 3x3). Matemáticamente, aplicar tres transformaciones lineales consecutivas es equivalente a aplicar una única transformación definida por el producto de sus matrices. Esta aproximación es computacionalmente más eficiente y elegante, ya que requiere una única operación de interpolación (remuestreo) sobre la imagen original, minimizando los artefactos de aliasing y el desenfoque acumulado. La multiplicación se realiza en orden inverso (Rotación * Cizallamiento * Escalado) para respetar el orden de aplicación de las operaciones lineales.

## Parte B2: Ingeniería Inversa de Transformaciones
Para este apartado, la estrategia ha sido deducir la matriz de transformación a partir de las restricciones geométricas finales conocidas, en lugar de por ensayo y error.

Sabemos que la distancia final entre puntos se reduce a 20, lo que implica un factor de escala de 0.5. El ángulo de inclinación de las líneas verticales nos da directamente el ángulo de cizallamiento (-30 grados). Con estos datos, construimos las matrices de escalado y rotación/cizallamiento.

La incógnita restante es la traslación. En lugar de adivinarla, se ha calculado analíticamente. Se aplica la transformación lineal (sin traslación) al centro original de la imagen y se compara la posición resultante con la posición deseada (100, 80). La diferencia entre ambas posiciones es el vector de traslación exacto que se debe incorporar a la matriz final.

## Parte C1: Transformación de Perspectiva
La deformación solicitada comprime la mitad izquierda de la imagen y expande la derecha. Esta transformación no conserva el paralelismo de las líneas verticales originales ni mantiene constantes las relaciones de distancia a lo largo del eje X.

Por definición, una transformación afín (warpAffine) debe preservar el paralelismo. Dado que la transformación requerida viola este principio, se justifica el uso de una homografía o transformación de perspectiva (warpPerspective). Solo una transformación proyectiva puede representar este "efecto de profundidad" donde la escala cambia progresivamente a lo largo de un eje. Se ha derivado la matriz de homografía imponiendo restricciones sobre cómo se mapean tres puntos clave del eje X (inicio, mitad, fin), lo que define unívocamente la distorsión no lineal requerida.

## Parte C2: Deformación No Lineal Genérica
La transformación especificada incluye un término cuadrático ($x^2$) en la ecuación de la coordenada Y. Las transformaciones afines y de perspectiva son, por definición, transformaciones lineales (en coordenadas homogéneas) y solo pueden representar planos y proyecciones rectas. No tienen capacidad para curvar líneas rectas en parábolas.

Por tanto, la única solución viable es utilizar un mapeo genérico píxel a píxel. Se ha decidido utilizar la función `remap` de OpenCV. Esta función permite definir explícitamente de dónde proviene cada píxel de la imagen destino mediante dos matrices de coordenadas (mapa X y mapa Y). Se ha implementado el mapeo inverso: para cada píxel de la imagen destino, calculamos su posición correspondiente en la imagen origen invirtiendo la fórmula dada. Esto garantiza que la imagen final no tenga "agujeros" (píxeles sin asignar), lo cual ocurriría si usáramos mapeo directo.
