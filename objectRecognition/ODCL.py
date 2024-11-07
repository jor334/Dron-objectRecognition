import cv2
from ultralytics import YOLO

# Cargar el modelo preentrenado YOLOv8 (revisa si tienes el archivo correcto en lugar de YOLOv10)
model = YOLO("models/yolov8s.pt")  # Asegúrate de tener el modelo correcto y disponible

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos en el frame
    results = model(frame, conf=0.3, verbose=False)

    # Verifica y accede a los resultados de detección
    for det in results[0].boxes:  # Accede a los cuadros delimitadores
        x1, y1, x2, y2 = det.xyxy[0]  # Coordenadas
        conf = det.conf[0]  # Confianza
        cls = det.cls[0]  # Clase detectada
        label = f"{model.names[int(cls)]}: {conf:.2f}"

        # Dibuja el cuadro y la etiqueta en el marco
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Muestra el frame con las detecciones
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
