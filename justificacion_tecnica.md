# Justificación Técnica de Procedimientos de Procesamiento de Imágenes
## Contexto: Digitalización y Mejora de Dibujos Escaneados

Este documento detalla y justifica las decisiones técnicas adoptadas en la implementación de la aplicación de procesamiento de imágenes. El objetivo principal es la mejora, restauración y análisis de **dibujos realizados a mano y posteriormente escaneados**. Este contexto específico presenta desafíos únicos como ruido del escáner, líneas tenues, iluminación desigual y artefactos de papel, que han guiado la selección de cada algoritmo.

---

## 1. Análisis Preliminar y Preprocesamiento

### 1.1. Selección de Región de Interés (ROI)
*   **Justificación**: Los escaneos a menudo incluyen bordes del escáner, sombras en los márgenes o áreas de papel vacío irrelevantes.
*   **Implementación**: Permite al usuario recortar la imagen antes de cualquier procesamiento.
*   **Beneficio**: Reduce el tiempo de cómputo al procesar solo el área relevante y evita que los bordes negros del escáner afecten a los cálculos del histograma o a la umbralización automática (Otsu).

### 1.2. Histogramas (Intensidades y 2D HSV)
*   **Justificación**: Antes de decidir qué filtro aplicar, es crucial entender la distribución de la luz y el contraste.
*   **Implementación**:
    *   **Histograma de Intensidades**: Permite visualizar si el dibujo tiene buen contraste o si está "lavado" (picos concentrados). Ayuda a elegir manualmente el valor de umbral.
    *   **Histograma 2D (HSV)**: Útil si el escaneo es en color, para analizar si hay dominantes de color (ej. papel amarillento) que se puedan separar de la tinta.

### 1.3. Estimación de Ruido Adaptativa
*   **Justificación**: No todos los escaneos son iguales. Una foto de un dibujo tiene ruido diferente (grano, ISO) que un escaneo binario (ruido de sal y pimienta).
*   **Implementación**:
    *   **Método MAD (Median Absolute Deviation)**: Se usa para imágenes con escala de grises/color para estimar la desviación estándar del ruido gaussiano de manera robusta.
    *   **Conteo de Píxeles Aislados**: Se activa automáticamente para imágenes binarias. Justificado porque en documentos binarizados, el "ruido" son puntos negros aislados o huecos blancos.

---

## 2. Filtrado y Reducción de Ruido

### 2.1. Filtro de Mediana (Median Filter)
*   **Justificación Crítica**: Los escáneres suelen introducir "ruido de sal y pimienta" (píxeles blancos o negros aleatorios) debido al polvo en el cristal o sensores defectuosos.
*   **Por qué este y no otro**: El filtro gaussiano suaviza este ruido pero lo "emborrona". El filtro de mediana es el **único** capaz de eliminar completamente estos píxeles ruidosos sin desenfocar los bordes de los trazos del dibujo.
*   **Ajuste**: Se ha implementado con un rango bajo (3-9) para evitar eliminar detalles finos del dibujo.

### 2.2. Filtro Bilateral
*   **Justificación**: Los dibujos a mano dependen de la nitidez de la línea.
*   **Por qué este y no otro**: A diferencia del desenfoque Gaussiano estándar que suaviza toda la imagen (difuminando las líneas del dibujo), el filtro Bilateral suaviza las texturas del papel (ruido) pero **preserva los bordes** (los trazos del lápiz/tinta). Es computacionalmente más costoso pero esencial para mantener la calidad del trazo.

### 2.3. Filtro Gaussiano
*   **Justificación**: Se mantiene como opción para reducción de ruido general de alta frecuencia (grano fino del papel) cuando la preservación estricta de bordes no es la prioridad máxima, o como paso previo a una umbralización muy agresiva.

---

## 3. Operaciones Morfológicas: Restauración y Limpieza
Esta es la sección más crítica para cumplir con el requisito de "restauración aproximada de líneas".

### 3.1. Dilatación (Restauración de Líneas)
*   **Problema**: Al escanear dibujos hechos con lápiz duro o trazos rápidos, algunas líneas quedan muy finas o discontinuas. Además, la binarización puede "romper" estas líneas.
*   **Solución**: La dilatación añade píxeles a los límites de los objetos (trazos negros sobre fondo blanco, o viceversa según la inversión).
*   **Justificación**: Permite **reconstruir y engrosar** las líneas que se hayan perdido o adelgazado durante el filtrado o umbralización, recuperando la continuidad del trazo original.

### 3.2. Erosión (Limpieza)
*   **Problema**: A veces el trazo del rotulador sangra en el papel, o la umbralización hace que las líneas sean demasiado gruesas.
*   **Solución**: La erosión elimina píxeles de los límites.
*   **Justificación**: Permite adelgazar trazos excesivamente gruesos y eliminar ruido pequeño que el filtro de mediana no haya captado.

### 3.3. Opening (Apertura)
*   **Justificación**: Combinación de Erosión seguida de Dilatación. Ideal para eliminar **ruido de fondo** (manchas pequeñas, polvo) sin alterar el tamaño general de los objetos grandes (el dibujo principal).

### 3.4. Closing (Cierre)
*   **Justificación**: Combinación de Dilatación seguida de Erosión. Fundamental para **cerrar pequeños huecos** dentro de las líneas del dibujo, haciendo que los trazos parezcan más sólidos y continuos sin engrosarlos excesivamente.

---

## 4. Segmentación y Binarización

### 4.1. Umbralización Manual (Varios Métodos)
*   **Justificación**: Permite al usuario tener control total cuando las condiciones de luz son inusuales. Se ofrecen variantes (Invertido, Truncado) para adaptarse a si el dibujo es tinta oscura sobre papel claro o viceversa (pizarras).

### 4.2. Método de Otsu (Automático)
*   **Justificación**: En muchos escaneos, encontrar el valor exacto de umbral (ej. 127) es prueba y error.
*   **Implementación**: El algoritmo de Otsu analiza el histograma bimodal (picos de papel y tinta) y calcula matemáticamente el umbral óptimo que minimiza la varianza intra-clase.
*   **Beneficio**: Proporciona una binarización óptima automática para la gran mayoría de documentos sin intervención del usuario.

---

## 5. Arquitectura del Flujo de Trabajo (Pipeline)

### 5.1. Procesamiento Secuencial Personalizable (Orden 1-12)
*   **Justificación**: No existe una "receta única" para restaurar un dibujo.
    *   A veces es mejor filtrar ruido -> luego umbralizar.
    *   Otras veces es mejor umbralizar -> luego limpiar con morfología.
*   **Implementación**: Se diseñó un sistema flexible donde el usuario define el orden. Esto es vital porque operaciones como la **Dilatación** deben aplicarse generalmente *después* de la umbralización para restaurar líneas, mientras que el **Filtro Bilateral** funciona mejor *antes* de la umbralización sobre la imagen en escala de grises.
