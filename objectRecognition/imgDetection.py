import cv2
import os
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO("models/yolov8m.pt")

# Ruta a la carpeta de imágenes
image_folder = "./dataExamples"

# Procesar cada imagen en la carpeta
for image_name in os.listdir(image_folder):
    # Crear la ruta completa a la imagen
    image_path = os.path.join(image_folder, image_name)
    
    # Leer la imagen
    image = cv2.imread(image_path)
    
    results = model(image)

    # Dibujar las detecciones en la imagen
    for det in results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0])  # Coordenadas de la caja delimitadora
        conf = det.conf[0]  # Confianza de la detección
        cls = int(det.cls[0])  # Clase detectada
        label = f"{model.names[cls]}: {conf:.2f}"

        # Dibuja el cuadro delimitador y la etiqueta en la imagen
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Muestra la imagen con las detecciones
    cv2.imshow("Detecciones YOLOv8", image)
    cv2.waitKey(0)  # Espera hasta que se presione una tecla para pasar a la siguiente imagen

# Cierra todas las ventanas al terminar
cv2.destroyAllWindows()
