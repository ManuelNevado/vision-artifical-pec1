import cv2 as cv
import numpy as np

def main():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return

    # Variables de estado
    track_window = None
    roi_hist = None
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    
    # Definir rectángulo inicial fijo (centro de la pantalla aprox)
    # x, y, w, h
    init_rect = (200, 200, 100, 100)
    tracking_initialized = False

    print("Instrucciones:")
    print("1. Coloca tu mano dentro del cuadrado azul.")
    print("2. Pulsa la tecla 'i' para iniciar el seguimiento.")
    print("3. Pulsa 'ESC' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espejo para que sea más natural
        frame = cv.flip(frame, 1)

        if tracking_initialized:
            # Convertir a HSV
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            # Calcular Back Projection
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            # Aplicar MeanShift
            ret, track_window = cv.meanShift(dst, track_window, term_crit)
            
            # Dibujar el resultado
            x, y, w, h = track_window
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, "Tracking", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        else:
            # Dibujar rectángulo de inicialización
            x, y, w, h = init_rect
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv.putText(frame, "Coloca la mano y pulsa 'i'", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv.imshow('MeanShift Hand Tracker', frame)

        k = cv.waitKey(30) & 0xff
        if k == 27: # ESC
            break
        elif k == ord('i') and not tracking_initialized:
            # Inicializar tracking
            x, y, w, h = init_rect
            roi = frame[y:y+h, x:x+w]
            hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            
            # Calcular máscara para ignorar brillos bajos/altos y saturación baja
            mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            
            # Calcular histograma (solo Hue)
            roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
            
            track_window = init_rect
            tracking_initialized = True
            print("Tracking iniciado.")

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
