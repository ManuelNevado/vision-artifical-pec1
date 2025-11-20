import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import io
import tempfile
import os


max_width = 500

def pil_to_cv(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    return cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)


def cv_to_pil(cv_image):
    """Convierte imagen OpenCV a formato PIL"""
    return Image.fromarray(cv.cvtColor(cv_image, cv.COLOR_BGR2RGB))


def select_roi(image, x1, y1, x2, y2):
    """
    Selecciona una región de interés (ROI) de una imagen.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
        x1, y1, x2, y2: Coordenadas de la ROI
    
    Returns:
        numpy.ndarray: Imagen recortada correspondiente a la ROI
    """
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def manual_threshold(image, threshold, mode=cv.THRESH_BINARY):
    """
    Aplica umbralización binaria manual a una imagen.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
        threshold: Valor del umbral (0-255)
        mode: Modo de umbralización (cv.THRESH_BINARY, cv.THRESH_BINARY_INV, etc.)
    
    Returns:
        numpy.ndarray: Imagen umbralizada
    """
    _, thresholded_image = cv.threshold(image, threshold, 255, mode)
    return thresholded_image


def histograma(image):
    """
    Calcula y visualiza el histograma 2D (Hue vs Saturation) de una imagen.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
    
    Returns:
        tuple: (histograma 2D, figura matplotlib)
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    histogram = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(histogram, interpolation='nearest')
    ax.set_title('Histograma 2D (Hue vs Saturation)')
    ax.set_xlabel('Saturation')
    ax.set_ylabel('Hue')
    plt.tight_layout()
    
    return histogram, fig


def hist_plot(image):
    """
    Calcula el histograma de intensidades de una imagen en escala de grises.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
    
    Returns:
        tuple: (count, r, figura matplotlib)
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    count = []
    r = []
    
    for k in range(256):
        r.append(k)
        count1 = np.sum(gray == k)
        count.append(count1)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(r, count, color='blue', alpha=0.7)
    ax.set_title('Histograma de Intensidades')
    ax.set_xlabel('Nivel de intensidad')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return count, r, fig


def gaussian_blur(image, kernel_size=5):
    """
    Aplica un filtro Gaussiano para reducir ruido.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
        kernel_size: Tamaño del kernel (debe ser impar, default: 5)
    
    Returns:
        numpy.ndarray: Imagen filtrada
    """
    # Asegurar que el kernel_size sea impar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def median_filter(image, kernel_size=5):
    """
    Aplica un filtro de mediana para eliminar ruido de sal y pimienta.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
        kernel_size: Tamaño del kernel (debe ser impar, default: 5)
    
    Returns:
        numpy.ndarray: Imagen filtrada
    """
    # Asegurar que el kernel_size sea impar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filtered = cv.medianBlur(image, kernel_size)
    return filtered


def morphological_opening(image, kernel_size=5):
    """
    Aplica operación morfológica de Opening (erosión + dilatación).
    Útil para eliminar ruido pequeño manteniendo los objetos grandes.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
        kernel_size: Tamaño del elemento estructurante (debe ser impar, default: 5)
    
    Returns:
        numpy.ndarray: Imagen después de aplicar Opening
    """
    # Asegurar que el kernel_size sea impar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Crear elemento estructurante rectangular
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    
    # Aplicar Opening (erosión seguida de dilatación)
    opened = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return opened


def morphological_closing(image, kernel_size=5):
    """
    Aplica operación morfológica de Closing (dilatación + erosión).
    Útil para rellenar huecos pequeños en objetos.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
        kernel_size: Tamaño del elemento estructurante (debe ser impar, default: 5)
    
    Returns:
        numpy.ndarray: Imagen después de aplicar Closing
    """
    # Asegurar que el kernel_size sea impar
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Crear elemento estructurante rectangular
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    
    # Aplicar Closing (dilatación seguida de erosión)
    closed = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    return closed


def estimate_noise(image):
    """
    Estima el nivel de ruido usando MAD (Median Absolute Deviation) 
    en diferencias de alta frecuencia.
    Este es un método robusto y estándar usado en literatura científica.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
    
    Returns:
        dict: Diccionario con métricas de ruido
            - noise_sigma: Estimación sigma del ruido (desviación estándar estimada)
            - noise_mad: MAD de las diferencias de alta frecuencia
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Convertir a float para cálculos precisos
    gray_float = gray.astype(np.float64)
    
    # Calcular diferencias de alta frecuencia (horizontal, vertical, diagonal)
    # Estos capturan el ruido sin ser afectados por bordes suaves
    diff_h = np.diff(gray_float, axis=1)  # Diferencias horizontales
    diff_v = np.diff(gray_float, axis=0)  # Diferencias verticales
    
    # Combinar todas las diferencias
    all_diffs = np.concatenate([diff_h.flatten(), diff_v.flatten()])
    
    # Calcular MAD (Median Absolute Deviation)
    # MAD es muy robusto a outliers (bordes fuertes)
    median = np.median(all_diffs)
    mad = np.median(np.abs(all_diffs - median))
    
    # Convertir MAD a estimación de sigma (desviación estándar)
    # Factor 1.4826 es el factor de escala para distribución normal
    sigma = mad * 1.4826
    
    return {
        'noise_sigma': sigma,
        'noise_mad': mad
    }




def main():
    """Función principal de la aplicación Streamlit"""
    
    # Configuración de la página
    st.set_page_config(
        page_title="Procesador de Imágenes",
        page_icon="�",
        layout="wide"
    )
    
    # Título y descripción
    st.title("Procesador de Imágenes")
    st.markdown("""
    Frontal ejercicio 1 PEC1.
    Formatos soportados: JPG, JPEG, PNG
    """)
    
    # Widget de carga de archivo
    uploaded_file = st.file_uploader(
        "Selecciona una imagen para procesar",
        type=["jpg", "jpeg", "png"],
        help="Sube una imagen en formato JPG, JPEG o PNG"
    )
    
    # Procesar si hay una imagen cargada
    if uploaded_file is not None:
        # Cargar la imagen
        try:
            image_pil = Image.open(uploaded_file)
            image_cv = pil_to_cv(image_pil)
            
            # Mostrar información de la imagen
            st.subheader("Información de la imagen")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Formato", image_pil.format if image_pil.format else "PNG")
            with col2:
                st.metric("Dimensiones", f"{image_pil.size[0]} x {image_pil.size[1]}")
            with col3:
                st.metric("Modo", image_pil.mode)
            
            st.divider()
            
            # Sidebar para seleccionar operaciones
            st.sidebar.header("Operaciones de Procesamiento")
            st.sidebar.markdown("Selecciona las operaciones que deseas aplicar:")
            
            # Checkboxes para cada operación con orden
            st.sidebar.markdown("**Selecciona y ordena las operaciones:**")
            st.sidebar.caption("Define el orden de ejecución (1 = primero, 6 = último)")
            
            operations = []
            
            # ROI
            col_cb, col_order = st.sidebar.columns([3, 1])
            with col_cb:
                use_roi = st.checkbox("ROI (Región de Interés)", key="cb_roi")
            with col_order:
                if use_roi:
                    order_roi = st.number_input("", min_value=1, max_value=6, value=1, key="order_roi", label_visibility="collapsed")
                    operations.append((order_roi, "roi"))
            
            # Filtro Gaussiano
            col_cb, col_order = st.sidebar.columns([3, 1])
            with col_cb:
                use_gaussian = st.checkbox("Filtro Gaussiano", key="cb_gaussian")
            with col_order:
                if use_gaussian:
                    order_gaussian = st.number_input("", min_value=1, max_value=6, value=2, key="order_gaussian", label_visibility="collapsed")
                    operations.append((order_gaussian, "gaussian"))
            
            # Filtro de Mediana
            col_cb, col_order = st.sidebar.columns([3, 1])
            with col_cb:
                use_median = st.checkbox("Filtro de Mediana", key="cb_median")
            with col_order:
                if use_median:
                    order_median = st.number_input("", min_value=1, max_value=8, value=3, key="order_median", label_visibility="collapsed")
                    operations.append((order_median, "median"))
            
            # Opening Morfológico
            col_cb, col_order = st.sidebar.columns([3, 1])
            with col_cb:
                use_opening = st.checkbox("Opening Morfológico", key="cb_opening")
            with col_order:
                if use_opening:
                    order_opening = st.number_input("", min_value=1, max_value=8, value=4, key="order_opening", label_visibility="collapsed")
                    operations.append((order_opening, "opening"))
            
            # Closing Morfológico
            col_cb, col_order = st.sidebar.columns([3, 1])
            with col_cb:
                use_closing = st.checkbox("Closing Morfológico", key="cb_closing")
            with col_order:
                if use_closing:
                    order_closing = st.number_input("", min_value=1, max_value=8, value=5, key="order_closing", label_visibility="collapsed")
                    operations.append((order_closing, "closing"))
            
            # Umbralización
            col_cb, col_order = st.sidebar.columns([3, 1])
            with col_cb:
                use_threshold = st.checkbox("Umbralización Manual", key="cb_threshold")
            with col_order:
                if use_threshold:
                    order_threshold = st.number_input("", min_value=1, max_value=8, value=6, key="order_threshold", label_visibility="collapsed")
                    operations.append((order_threshold, "threshold"))
            
            # Histograma 2D
            col_cb, col_order = st.sidebar.columns([3, 1])
            with col_cb:
                use_hist_2d = st.checkbox("Histograma 2D (HSV)", key="cb_hist2d")
            with col_order:
                if use_hist_2d:
                    order_hist_2d = st.number_input("", min_value=1, max_value=8, value=7, key="order_hist_2d", label_visibility="collapsed")
                    operations.append((order_hist_2d, "hist_2d"))
            
            # Histograma de Intensidades
            col_cb, col_order = st.sidebar.columns([3, 1])
            with col_cb:
                use_hist_plot = st.checkbox("Histograma Intensidades", key="cb_histplot")
            with col_order:
                if use_hist_plot:
                    order_hist_plot = st.number_input("", min_value=1, max_value=8, value=8, key="order_hist_plot", label_visibility="collapsed")
                    operations.append((order_hist_plot, "hist_plot"))
            
            # Ordenar operaciones según prioridad
            operations.sort(key=lambda x: x[0])
            operation_order = [op[1] for op in operations]
            
            st.sidebar.divider()
            
            # Parámetros para cada operación
            roi_params = None
            threshold_value = None
            
            if use_roi:
                st.sidebar.subheader("Parámetros ROI")
                x1 = st.sidebar.slider("X1 (inicio)", 0, image_pil.size[0], 0)
                y1 = st.sidebar.slider("Y1 (inicio)", 0, image_pil.size[1], 0)
                x2 = st.sidebar.slider("X2 (fin)", 0, image_pil.size[0], min(200, image_pil.size[0]))
                y2 = st.sidebar.slider("Y2 (fin)", 0, image_pil.size[1], min(200, image_pil.size[1]))
                roi_params = (x1, y1, x2, y2)
            
            if use_gaussian:
                st.sidebar.subheader("Parámetros Filtro Gaussiano")
                gaussian_kernel = st.sidebar.slider("Tamaño del kernel", 3, 15, 5, step=2,
                                                   help="Kernel más grande = mayor suavizado")
            else:
                gaussian_kernel = None
            
            if use_median:
                st.sidebar.subheader("Parámetros Filtro Mediana")
                median_kernel = st.sidebar.slider("Tamaño del kernel", 3, 15, 5, step=2,
                                                 help="Efectivo para ruido sal y pimienta")
            else:
                median_kernel = None
            
            if use_opening:
                st.sidebar.subheader("Parámetros Opening")
                opening_kernel = st.sidebar.slider("Tamaño del kernel", 3, 15, 5, step=2,
                                                   help="Elimina ruido pequeño", key="opening_kernel")
            else:
                opening_kernel = None
            
            if use_closing:
                st.sidebar.subheader("Parámetros Closing")
                closing_kernel = st.sidebar.slider("Tamaño del kernel", 3, 15, 5, step=2,
                                                   help="Rellena huecos pequeños", key="closing_kernel")
            else:
                closing_kernel = None
            
            if use_threshold:
                st.sidebar.subheader("Parámetros Umbralización")
                threshold_value = st.sidebar.slider("Valor del umbral", 0, 255, 127)
            else:
                threshold_value = None
            
            # Mostrar imagen original
            st.subheader("Imagen Original")
            st.image(image_pil, width=max_width)
            
            st.divider()
            
            # Botón para procesar
            if st.button("Procesar Imagen", type="primary", use_container_width=True):
                with st.spinner("Procesando imagen..."):
                    results = []
                    
                    # Imagen de trabajo - comienza con la imagen original
                    working_image = image_cv.copy()
                    
                    st.subheader("Pipeline de Procesamiento")
                    if operation_order:
                        st.caption(f"Orden de ejecución: {' → '.join([op.upper() for op in operation_order])}")
                    st.markdown("---")
                    
                    # Punto de referencia para análisis de ruido (después de ROI si existe)
                    reference_image = None
                    noise_before = None
                    
                    # Ejecutar operaciones según el orden especificado
                    for operation in operation_order:
                        paso_num = len(results) + 1
                        
                        if operation == "roi" and roi_params:
                            st.markdown(f"**Paso {paso_num}: ROI (Región de Interés)**")
                            working_image = select_roi(working_image, *roi_params)
                            st.image(cv_to_pil(working_image), caption=f"ROI aplicada: ({roi_params[0]}, {roi_params[1]}) a ({roi_params[2]}, {roi_params[3]})", width=max_width)
                            results.append("ROI")
                            st.markdown("---")
                            
                            # Actualizar referencia después de ROI
                            if reference_image is None:
                                reference_image = working_image.copy()
                        
                        elif operation == "gaussian" and gaussian_kernel is not None:
                            # Calcular ruido antes si es el primer filtro
                            if reference_image is None:
                                reference_image = working_image.copy()
                            if noise_before is None:
                                noise_before = estimate_noise(reference_image)
                            
                            st.markdown(f"**Paso {paso_num}: Filtro Gaussiano**")
                            working_image = gaussian_blur(working_image, gaussian_kernel)
                            st.image(cv_to_pil(working_image), caption=f"Filtro Gaussiano aplicado (kernel={gaussian_kernel}x{gaussian_kernel})", width=max_width)
                            results.append("Filtro Gaussiano")
                            st.markdown("---")
                        
                        elif operation == "median" and median_kernel is not None:
                            # Calcular ruido antes si es el primer filtro
                            if reference_image is None:
                                reference_image = working_image.copy()
                            if noise_before is None:
                                noise_before = estimate_noise(reference_image)
                            
                            st.markdown(f"**Paso {paso_num}: Filtro de Mediana**")
                            working_image = median_filter(working_image, median_kernel)
                            st.image(cv_to_pil(working_image), caption=f"Filtro de Mediana aplicado (kernel={median_kernel}x{median_kernel})", width=max_width)
                            results.append("Filtro Mediana")
                            st.markdown("---")
                        
                        elif operation == "opening" and opening_kernel is not None:
                            st.markdown(f"**Paso {paso_num}: Opening Morfológico**")
                            working_image = morphological_opening(working_image, opening_kernel)
                            st.image(cv_to_pil(working_image), caption=f"Opening aplicado (kernel={opening_kernel}x{opening_kernel})", width=max_width)
                            results.append("Opening")
                            st.markdown("---")
                        
                        elif operation == "closing" and closing_kernel is not None:
                            st.markdown(f"**Paso {paso_num}: Closing Morfológico**")
                            working_image = morphological_closing(working_image, closing_kernel)
                            st.image(cv_to_pil(working_image), caption=f"Closing aplicado (kernel={closing_kernel}x{closing_kernel})", width=max_width)
                            results.append("Closing")
                            st.markdown("---")
                        
                        elif operation == "threshold" and threshold_value is not None:
                            st.markdown(f"**Paso {paso_num}: Umbralización Manual**")
                            working_image = manual_threshold(working_image, threshold_value)
                            st.image(cv_to_pil(working_image), caption=f"Umbralización aplicada con valor {threshold_value}", width=max_width)
                            results.append("Umbralización")
                            st.markdown("---")
                        
                        elif operation == "hist_2d":
                            st.markdown(f"**Paso {paso_num}: Histograma 2D (Hue vs Saturation)**")
                            st.caption("Calculado sobre la imagen procesada hasta este punto")
                            hist_2d, fig_2d = histograma(working_image)
                            st.pyplot(fig_2d)
                            plt.close(fig_2d)
                            results.append("Histograma 2D")
                            st.markdown("---")
                        
                        elif operation == "hist_plot":
                            st.markdown(f"**Paso {paso_num}: Histograma de Intensidades**")
                            st.caption("Calculado sobre la imagen procesada hasta este punto")
                            count, r, fig_plot = hist_plot(working_image)
                            st.pyplot(fig_plot)
                            plt.close(fig_plot)
                            results.append("Histograma de Intensidades")
                            st.markdown("---")
                    
                    # Análisis de ruido si se aplicaron filtros
                    if noise_before is not None and (use_gaussian or use_median):
                        st.subheader("Análisis de Ruido")
                        
                        # Calcular ruido después de aplicar filtros
                        noise_after = estimate_noise(working_image)
                        
                        # Calcular reducción de ruido
                        reduction_sigma = ((noise_before['noise_sigma'] - noise_after['noise_sigma']) / noise_before['noise_sigma']) * 100
                        
                        # Mostrar métricas en columnas
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Ruido Inicial",
                                f"{noise_before['noise_sigma']:.2f}",
                                help="Estimación sigma del ruido (mayor = más ruido)"
                            )
                        
                        with col2:
                            st.metric(
                                "Ruido Final",
                                f"{noise_after['noise_sigma']:.2f}",
                                delta=f"{-reduction_sigma:.1f}%",
                                delta_color="inverse",
                                help="Reducción de ruido aplicada"
                            )
                        
                        with col3:
                            improvement = "Mejora" if reduction_sigma > 0 else "Aumento"
                            st.metric(
                                f"{improvement} de Calidad",
                                f"{abs(reduction_sigma):.1f}%",
                                help="Porcentaje de reducción de ruido"
                            )
                        
                        # Nota informativa
                        if use_roi:
                            st.caption("Análisis calculado sobre la región ROI seleccionada")
                        else:
                            st.caption("Análisis calculado sobre la imagen completa")
                        
                        st.markdown("---")
                    
                    # Mensaje de éxito
                    if results:
                        st.success(f"Procesamiento completado. Pipeline ejecutado: {' → '.join(results)}")
                        
                        # Mostrar imagen final si hubo transformaciones
                        if use_roi or use_threshold or use_gaussian or use_median:
                            st.subheader("Imagen Final Procesada")
                            st.image(cv_to_pil(working_image), caption="Resultado final después de aplicar todas las transformaciones", width=max_width)
                    else:
                        st.warning("No se seleccionó ninguna operación. Por favor, marca al menos una casilla en la barra lateral.")
        
        except Exception as e:
            st.error(f"Error al cargar la imagen: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        # Mensaje cuando no hay imagen cargada
        st.info("Sube una imagen usando el botón de arriba para comenzar")


if __name__ == "__main__":
    main()
