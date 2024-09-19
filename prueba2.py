import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo YOLO personalizado
model = YOLO("best.pt")

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Leer el frame de la cámara

    if not ret:
        break

    # Realizar la inferencia sobre el frame
    results = model(frame)  # Se pasan las imágenes directamente al modelo

    # Iterar sobre cada detección en los resultados
    for result in results:
        for detection in result.boxes:  # Asegurarse de acceder a las detecciones de forma correcta
            # Extraer las coordenadas de la caja delimitadora
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Coordenadas (x1, y1) y (x2, y2)
            
            # Recortar la región de interés (ROI) para obtener el color
            roi = frame[y1:y2, x1:x2]
            
            # Convertir a HSV para detectar el color
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Calcular el color promedio en la región de interés
            avg_color_per_row = np.average(hsv, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)

            hue_value = avg_color[0]  # El valor de tono (hue) en HSV indica el color

            # Clasificar el color basado en el valor de hue
            if hue_value < 15 or hue_value > 160:
                color = "Rojo"
                color_bgr = (0, 0, 255)  # Rojo en formato BGR
            elif 15 <= hue_value < 35:
                color = "Amarillo"
                color_bgr = (0, 255, 255)  # Amarillo en BGR
            elif 35 <= hue_value < 85:
                color = "Verde"
                color_bgr = (0, 255, 0)  # Verde en BGR
            elif 85 <= hue_value < 130:
                color = "Azul"
                color_bgr = (255, 0, 0)  # Azul en BGR
            else:
                color = "Otro color"
                color_bgr = (255, 255, 255)  # Blanco para otros colores

            # Extraer la etiqueta de clase (nombre del objeto)
            label = detection.cls[0]
            confidence = detection.conf[0]  # Extraer la confianza de la detección

            # Dibujar la caja delimitadora en el color identificado
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        
            # Colocar el nombre del objeto, color y la confianza en la imagen
            text = f"{label}, color: {color}, confianza: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

    # Mostrar la imagen con las detecciones
    cv2.imshow('Detections', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()