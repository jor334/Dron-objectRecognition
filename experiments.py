import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Cargar el modelo YOLO personalizado
model = YOLO("best.pt")

# Iniciar la captura de video


figAI = model("test.png")

for fig in figAI:
    for detection in fig.boxes:
        # x1, y1, x2, y2 = map(int, detection.xyxy[0])
        # roi = fig[y1:y2, x1:x2]

        class_id = int(detection.cls[0])
        label = model.names[class_id]

        ocr = easyocr.Reader(['en'])
        result = ocr.readtext('test.png')
        for x in result:
            letter = x[1]  # Imprimir el texto detectado

print(label, letter)
