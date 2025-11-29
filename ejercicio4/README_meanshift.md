# Seguimiento de Mano con MeanShift

## Descripción General
Esta aplicación utiliza el algoritmo MeanShift para realizar el seguimiento de una mano u otro objeto de color distintivo a través de la webcam. El objetivo es demostrar el uso de histogramas de color y retroproyección para el tracking en tiempo real.

## Requisitos del Sistema
Para ejecutar esta aplicación es necesario disponer de un entorno con Python 3 instalado. Además, se requiere la instalación de las librerías OpenCV (paquete opencv-python) y NumPy para el procesamiento de imágenes y cálculos matriciales.

## Instrucciones de Ejecución
La aplicación se debe lanzar desde la línea de comandos. Abra una terminal en la carpeta raíz del proyecto y ejecute el script mediante el comando python ejercicio4/meanshift_hand.py. Esto iniciará la captura de vídeo y abrirá la ventana principal de la aplicación.

## Proceso de Inicialización
Al iniciar el programa, aparecerá una ventana mostrando la señal de la webcam. En el centro de la imagen se visualizará un rectángulo fijo de color azul. El usuario debe colocar su mano o el objeto que desea seguir dentro de este recuadro, asegurándose de que ocupe la mayor parte posible del área para obtener una muestra de color limpia. Una vez posicionado el objeto, se debe pulsar la tecla i para capturar el modelo de color e iniciar el proceso de seguimiento.

## Comportamiento del Seguimiento
Tras la inicialización, el rectángulo cambiará su color a verde y dejará de estar fijo en el centro. A partir de este momento, el recuadro se ajustará dinámicamente a la posición del objeto detectado. El usuario puede mover la mano libremente por la pantalla y observar cómo el algoritmo actualiza la posición de la ventana de seguimiento en cada fotograma.

## Finalización
Para cerrar la aplicación y liberar la cámara, el usuario debe pulsar la tecla ESC en cualquier momento.

## Funcionamiento Técnico
El funcionamiento interno del script se basa en una secuencia de procesamiento de imagen. Primero, al pulsar la tecla de inicio, se extrae la Región de Interés (ROI) delimitada por el rectángulo inicial. Esta región se convierte del espacio de color BGR al espacio HSV, el cual es más robusto frente a cambios de iluminación.

A continuación, se calcula el histograma del canal Hue (Matiz) de esta región, generando una huella digital del color del objeto. En cada nuevo fotograma capturado por la cámara, se calcula la retroproyección del histograma. Este proceso crea una imagen de probabilidad donde cada píxel tiene un valor proporcional a la similitud de su color con el modelo capturado.

Finalmente, se aplica el algoritmo MeanShift sobre esta imagen de retroproyección. Este algoritmo busca iterativamente el pico de densidad de probabilidad, desplazando la ventana de seguimiento hacia el nuevo centroide del objeto en cada iteración, logrando así un seguimiento fluido.
