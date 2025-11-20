import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import io
import tempfile
import os


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


def manual_threshold(image, threshold):
    """
    Aplica umbralización binaria manual a una imagen.
    
    Args:
        image: Imagen en formato OpenCV (numpy.ndarray)
        threshold: Valor del umbral (0-255)
    
    Returns:
        numpy.ndarray: Imagen umbralizada
    """
    _, thresholded_image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
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
            
            # Checkboxes para cada operación
            use_roi = st.sidebar.checkbox("Seleccionar ROI (Región de Interés)")
            use_threshold = st.sidebar.checkbox("Umbralización Manual")
            use_hist_2d = st.sidebar.checkbox("Histograma 2D (HSV)")
            use_hist_plot = st.sidebar.checkbox("Histograma de Intensidades")
            
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
            
            if use_threshold:
                st.sidebar.subheader("Parámetros Umbralización")
                threshold_value = st.sidebar.slider("Valor del umbral", 0, 255, 127)
            
            # Mostrar imagen original
            st.subheader("Imagen Original")
            st.image(image_pil, use_container_width=True)
            
            st.divider()
            
            # Botón para procesar
            if st.button("Procesar Imagen", type="primary", use_container_width=True):
                with st.spinner("Procesando imagen..."):
                    results = []
                    
                    # Imagen de trabajo - comienza con la imagen original
                    working_image = image_cv.copy()
                    
                    st.subheader("Pipeline de Procesamiento")
                    st.markdown("---")
                    
                    # Aplicar ROI si está seleccionado
                    if use_roi and roi_params:
                        st.markdown("**Paso 1: ROI (Región de Interés)**")
                        working_image = select_roi(working_image, *roi_params)
                        st.image(cv_to_pil(working_image), caption=f"ROI aplicada: ({roi_params[0]}, {roi_params[1]}) a ({roi_params[2]}, {roi_params[3]})", use_container_width=True)
                        results.append("ROI")
                        st.markdown("---")
                    
                    # Aplicar umbralización si está seleccionado
                    if use_threshold and threshold_value is not None:
                        paso_num = len(results) + 1
                        st.markdown(f"**Paso {paso_num}: Umbralización Manual**")
                        working_image = manual_threshold(working_image, threshold_value)
                        st.image(cv_to_pil(working_image), caption=f"Umbralización aplicada con valor {threshold_value}", use_container_width=True)
                        results.append("Umbralización")
                        st.markdown("---")
                    
                    # Calcular histograma 2D si está seleccionado (sobre la imagen procesada)
                    if use_hist_2d:
                        paso_num = len(results) + 1
                        st.markdown(f"**Paso {paso_num}: Histograma 2D (Hue vs Saturation)**")
                        st.caption("Calculado sobre la imagen procesada hasta este punto")
                        hist_2d, fig_2d = histograma(working_image)
                        st.pyplot(fig_2d)
                        plt.close(fig_2d)
                        results.append("Histograma 2D")
                        st.markdown("---")
                    
                    # Calcular histograma de intensidades si está seleccionado (sobre la imagen procesada)
                    if use_hist_plot:
                        paso_num = len(results) + 1
                        st.markdown(f"**Paso {paso_num}: Histograma de Intensidades**")
                        st.caption("Calculado sobre la imagen procesada hasta este punto")
                        count, r, fig_plot = hist_plot(working_image)
                        st.pyplot(fig_plot)
                        plt.close(fig_plot)
                        results.append("Histograma de Intensidades")
                        st.markdown("---")
                    
                    # Mensaje de éxito
                    if results:
                        st.success(f"Procesamiento completado. Pipeline ejecutado: {' → '.join(results)}")
                        
                        # Mostrar imagen final si hubo transformaciones
                        if use_roi or use_threshold:
                            st.subheader("Imagen Final Procesada")
                            st.image(cv_to_pil(working_image), caption="Resultado final después de aplicar todas las transformaciones", use_container_width=True)
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
