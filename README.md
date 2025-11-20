# PEC1 Vision Artificial - Análisis y Procesamiento de Imágenes

Aplicación web interactiva desarrollada con Streamlit para el procesamiento y análisis de imágenes utilizando OpenCV.

## Descripción

Esta aplicación permite realizar operaciones de procesamiento de imagen de forma secuencial, donde cada operación se aplica sobre el resultado de la operación anterior, creando un pipeline de procesamiento configurable. El usuario puede definir el orden de ejecución de las operaciones según sus necesidades.

## Características

### Operaciones de Procesamiento

La aplicación incluye las siguientes operaciones de procesamiento de imágenes:

1.  **Selección de ROI (Región de Interés)**
    -   Selecciona y extrae una región específica de la imagen
    -   Control mediante sliders para coordenadas X1, Y1, X2, Y2

2.  **Filtro Gaussiano**
    -   Reduce ruido mediante desenfoque Gaussiano
    -   Tamaño de kernel ajustable (3-15 píxeles)
    -   Ideal para suavizar imágenes

3.  **Filtro de Mediana**
    -   Elimina ruido de sal y pimienta
    -   Tamaño de kernel ajustable (3-15 píxeles)
    -   Preserva bordes mejor que el filtro Gaussiano

4.  **Opening Morfológico**
    -   Erosión seguida de dilatación
    -   Elimina ruido pequeño manteniendo objetos grandes
    -   Tamaño de kernel ajustable (3-15 píxeles)
    -   Útil para limpiar puntos blancos aislados

5.  **Closing Morfológico**
    -   Dilatación seguida de erosión
    -   Rellena huecos pequeños en objetos
    -   Tamaño de kernel ajustable (3-15 píxeles)
    -   Útil para cerrar pequeñas discontinuidades

6.  **Umbralización Manual**
    -   Convierte imagen según diferentes métodos de umbralización
    -   Valor de umbral ajustable (0-255)
    -   5 métodos disponibles:
        - **Binary**: Píxeles > umbral → 255, píxeles ≤ umbral → 0
        - **Binary Invertido**: Píxeles > umbral → 0, píxeles ≤ umbral → 255
        - **Truncado**: Píxeles > umbral → valor umbral, píxeles ≤ umbral → mantiene valor
        - **A Cero**: Píxeles > umbral → mantiene valor, píxeles ≤ umbral → 0
        - **A Cero Invertido**: Píxeles > umbral → 0, píxeles ≤ umbral → mantiene valor
    -   Útil para segmentación y binarización

7.  **Dilatación**
    -   Operación morfológica básica
    -   Engrosa los objetos blancos
    -   Tamaño de kernel ajustable (3-15 píxeles)
    -   **Esencial para restaurar líneas adelgazadas por filtros**

8.  **Erosión**
    -   Operación morfológica básica
    -   Adelgaza los objetos blancos
    -   Tamaño de kernel ajustable (3-15 píxeles)
    -   Útil para eliminar ruido conectado

9.  **Filtro Bilateral**
    -   Reduce ruido preservando bordes nítidos
    -   Parámetros ajustables (diámetro y sigma)
    -   **Ideal para documentos escaneados** - mantiene las líneas definidas

10. **Gradient Morfológico**
    -   Detecta contornos (dilatación - erosión)
    -   Tamaño de kernel ajustable (3-15 píxeles)
    -   Útil para análisis de bordes

11. **Histograma 2D (HSV)**
    -   Calcula y muestra histograma 2D en espacio de color HSV
    -   Representa niveles de Hue (tono) vs Saturation (saturación)

12. **Histograma de Intensidades**
    -   Calcula distribución de intensidades de píxeles
    -   Muestra frecuencia de cada nivel (0-255)
    -   Calculado en escala de grises

### Pipeline de Procesamiento Secuencial Personalizable

**Novedad**: Las operaciones se ejecutan en el **orden que el usuario defina** mediante controles numéricos (1-12). Cada operación procesa el resultado de la operación anterior según la secuencia especificada.

**Ejemplo de pipeline personalizado:**
```
Imagen Original → ROI (orden 1) → Filtro Mediana (orden 2) → Dilatación (orden 3) → Umbralización (orden 4)
```

Los histogramas se calculan sobre la imagen procesada hasta ese punto, no sobre la imagen original.

## Instalación

### Requisitos Previos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instrucciones

1. Clonar o descargar el repositorio

2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Las dependencias incluyen:
- `streamlit` - Framework web para la interfaz
- `opencv-python` - Procesamiento de imágenes
- `matplotlib` - Visualización de histogramas
- `numpy` - Operaciones numéricas
- `Pillow` - Manejo de imágenes

## Uso

### Ejecutar la Aplicación

```bash
streamlit run app_streamlit.py
```

La aplicación se abrirá automáticamente en el navegador en `http://localhost:8501`

### Flujo de Trabajo

1. **Subir Imagen**
   - Clic en "Browse files" o arrastrar una imagen
   - Formatos soportados: JPG, JPEG, PNG

2. **Seleccionar Operaciones**
   - Marcar checkboxes en la barra lateral para las operaciones deseadas
   - Las operaciones se aplicarán en el orden seleccionado

3. **Configurar Parámetros**
   - Ajustar sliders según la operación seleccionada
   - Los parámetros aparecen solo para operaciones activas

4. **Procesar**
   - Clic en "Procesar Imagen"
   - Ver resultados paso a paso

5. **Analizar Resultados**
   - Cada paso muestra la imagen procesada hasta ese punto
   - Los histogramas reflejan la imagen acumulada
   - Revisa el análisis de ruido si aplicaste filtros

6. **Guardar Imagen (Opcional)**
   - Clic en "Guardar Imagen Procesada" al final del procesamiento
   - La imagen se guarda en la carpeta `resultados/`
   - El nombre incluye las operaciones aplicadas

## Estructura del Proyecto

```
PEC1/
├── app_streamlit.py      # Aplicación principal
├── ejercicio1.py         # Funciones originales de procesamiento
├── requirements.txt      # Dependencias del proyecto
├── README.md            # Este archivo
└── resultados/          # Carpeta para imágenes procesadas (se crea automáticamente)
```

## Funciones Principales

### Conversión de Formatos
- `pil_to_cv(pil_image)` - Convierte PIL a OpenCV
- `cv_to_pil(cv_image)` - Convierte OpenCV a PIL

### Procesamiento
- `select_roi(image, x1, y1, x2, y2)` - Extrae región de interés
- `gaussian_blur(image, kernel_size)` - Aplica filtro Gaussiano
- `median_filter(image, kernel_size)` - Aplica filtro de mediana
- `morphological_opening(image, kernel_size)` - Aplica Opening morfológico
- `morphological_closing(image, kernel_size)` - Aplica Closing morfológico
- `morphological_dilation(image, kernel_size)` - Aplica dilatación (restaura líneas)
- `morphological_erosion(image, kernel_size)` - Aplica erosión (adelgaza líneas)
- `bilateral_filter(image, d, sigma_color, sigma_space)` - Filtro bilateral (preserva bordes)
- `morphological_gradient(image, kernel_size)` - Aplica gradient morfológico
- `manual_threshold(image, threshold, mode)` - Umbralización binaria
- `histograma(image)` - Calcula histograma 2D HSV
- `hist_plot(image)` - Calcula histograma de intensidades
- `estimate_noise(image)` - Estima nivel de ruido (adaptativo según tipo de imagen)

### Análisis de Ruido

La aplicación incluye un **sistema automático de análisis de ruido adaptativo** que se activa cuando se utilizan filtros de reducción de ruido (Gaussiano o Mediana).

**Método de estimación adaptativo:**

**Para imágenes binarias (documentos escaneados):**
- Detecta automáticamente cuando la imagen tiene ≤10 valores únicos
- Cuenta píxeles aislados (ruido de sal y pimienta)
- Aplica Opening morfológico para identificar imperfecciones
- Calcula porcentaje de píxeles ruidosos
- Muestra resultados con símbolo %

**Para imágenes con gradientes (fotografías):**
- Utiliza **MAD (Median Absolute Deviation)** aplicado a diferencias de alta frecuencia
- Calcula diferencias entre píxeles adyacentes (horizontal y vertical)
- La mediana proporciona robustez ante bordes fuertes y outliers
- Convierte MAD a sigma (desviación estándar) usando el factor estadístico 1.4826
- Método estándar en literatura científica de procesamiento de imágenes

**Funcionamiento:**
1. **Punto de referencia**: Si hay ROI seleccionada, el análisis toma como referencia la imagen después de aplicar el ROI. Si no, usa la imagen original.
2. **Cálculo antes/después**: Estima el ruido antes de aplicar filtros y después de aplicarlos
3. **Métricas mostradas**:
   - Ruido Inicial (% para binarias, sigma para gradientes)
   - Ruido Final (con porcentaje de reducción)
   - Mejora de Calidad (porcentaje total)

**Ventajas del método adaptativo**:
- Funciona correctamente en documentos escaneados binarios
- Robusto ante bordes definidos (importante después de umbralización)
- No se afecta por valores extremos en la imagen
- Captura únicamente ruido, no estructura de la imagen
- Mejor correlación con la percepción visual del ruido
- Funciona correctamente en pipeline: ROI → Filtros → Umbralización


## Tecnologías Utilizadas

- **Streamlit** - Framework web interactivo
- **OpenCV** - Biblioteca de visión por computador
- **Matplotlib** - Visualización de datos
- **NumPy** - Cálculos numéricos
- **Pillow** - Procesamiento de imágenes

## Autor

Desarrollado para la asignatura de Visión Artificial (PEC1)
Master en Investigación en Inteligencia Artificial - UNED

## Notas Técnicas

- El tamaño máximo de visualización de imágenes está configurado a 500px de ancho
- Los kernels de filtrado deben ser impares (se ajustan automáticamente)
- El histograma 2D utiliza el espacio de color HSV para mejor representación
- Las imágenes se procesan internamente en formato BGR (OpenCV)
- Las imágenes guardadas se almacenan en la carpeta `resultados/` con nombre descriptivo
- El nombre de archivo guardado incluye todas las operaciones aplicadas

## Ejemplos de Uso

### Limpieza de Ruido
1. Seleccionar "Filtro Gaussiano" o "Filtro de Mediana"
2. Ajustar kernel según el nivel de ruido
3. Procesar para ver resultado

### Segmentación de Imagen
1. Aplicar filtros de ruido (opcional)
2. Seleccionar "Umbralización Manual"
3. Ajustar threshold hasta obtener segmentación deseada

### Análisis de Región Específica
1. Seleccionar "ROI"
2. Ajustar coordenadas para la región de interés
3. Aplicar operaciones posteriores solo en esa región
4. Ver histogramas específicos de la región

### Limpieza de Documentos Escaneados
1. Seleccionar "Filtro de Mediana" (orden 1)
2. Aplicar "Opening Morfológico" (orden 2) para eliminar puntos aislados
3. Aplicar "Closing Morfológico" (orden 3) para cerrar líneas discontinuas
4. Opcionalmente aplicar "Umbralización" (orden 4) para binarizar
5. Revisar análisis de ruido automático
6. Guardar imagen procesada

## Licencia

Este proyecto es material académico para fines educativos.
