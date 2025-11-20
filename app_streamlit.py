import streamlit as st
from PIL import Image
import numpy as np


def process_image(image):
    """
    Función placeholder para procesamiento de imágenes.
    Por ahora retorna la imagen sin modificaciones.
    
    Args:
        image: Imagen PIL
        
    Returns:
        Imagen PIL sin modificar
    """
    # TODO: Implementar procesamiento real de la imagen
    return image


def main():
    """Función principal de la aplicación Streamlit"""
    
    # Configuración de la página
    st.set_page_config(
        page_title="Procesador de Imágenes",
        page_icon="",
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
            image = Image.open(uploaded_file)
            
            # Mostrar información de la imagen
            st.subheader("Información de la imagen")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Formato", image.format)
            with col2:
                st.metric("Dimensiones", f"{image.size[0]} x {image.size[1]}")
            with col3:
                st.metric("Modo", image.mode)
            
            # Dividir en dos columnas para mostrar original y procesada
            col_left, col_right = st.columns(2)
            
            # Mostrar imagen original
            with col_left:
                st.subheader("Imagen Original")
                st.image(image, use_container_width=True)
            
            # Botón para procesar
            if st.button("Procesar Imagen", type="primary"):
                with st.spinner("Procesando imagen..."):
                    # Procesar la imagen (por ahora es un placeholder)
                    processed_image = process_image(image)
                    
                    # Mostrar imagen procesada
                    with col_right:
                        st.subheader("Imagen Procesada")
                        st.image(processed_image, use_container_width=True)
                    
                    st.success("Imagen procesada correctamente!")
        
        except Exception as e:
            st.error(f"Error al cargar la imagen: {str(e)}")
    else:
        # Mensaje cuando no hay imagen cargada
        st.info("👆 Sube una imagen usando el botón de arriba para comenzar")


if __name__ == "__main__":
    main()
