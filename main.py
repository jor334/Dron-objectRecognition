import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

# Cargar el modelo YOLO personalizado
model = YOLO("best.pt")


targets = [
    { "name": "Circle", "color": "Azul", "letter": "A" },
    { "name": "Square", "color": "Rojo", "letter": "B" },
    { "name": "Triangle", "color": "Verde", "letter": "C" },
]

#------------------------------funcion de color-------------------------------------
def get_color_name(bgr):
    colors = {
        "Rojo": [0, 0, 255],
        "Verde": [0, 255, 0],
        "Azul": [255, 0, 0],
        "Amarillo": [0, 255, 255],
        "Gris": [128, 128, 128],
        "Negro": [0, 0, 0],
        "Blanco": [255, 255, 255],
        "Naranja": [0, 165, 255],  
        "Violeta": [128, 0, 128],   
        "Marron": [42, 42, 165]     
    }
    
    min_distance = float('inf')
    color_name = "Desconocido"
    
    for name, color in colors.items():
        distance = np.linalg.norm(np.array(bgr) - np.array(color))
        if distance < min_distance:
            min_distance = distance
            color_name = name
            
    return color_name

def detectarColor(roi):

    frame = cv2.resize(roi, (640, 480))

    # Recortar el centro de la imagen
    height, width, _ = frame.shape
    center_region = frame[int(height/4):int(3*height/4), int(width/4):int(3*width/4)]

    # Calcular el histograma de la región central
    histogram = cv2.calcHist([center_region], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()

    # Suavizar el histograma (opcional)
    histogram = cv2.bilateralFilter(histogram, 9, 75, 75)

    # Encontrar el índice del color más frecuente
    max_index = np.argmax(histogram)
    blue_index, green_index, red_index = np.unravel_index(max_index, (8, 8, 8))

    # Convertir el índice a valores BGR
    predominant_color = np.array([blue_index * 32, green_index * 32, red_index * 32], dtype=np.uint8)

    # Obtener el nombre del color predominante
    color_name = get_color_name(predominant_color)
    return color_name

#------------------------------funcion OCR----------------------------------------------
def OCR(roi):
    ocr = easyocr.Reader(['en'])
    result = ocr.readtext(roi)
    for x in result:
        letter = x[1]  # Imprimir el texto detectado
        return letter
    

#------------------------------CORE-----------------------------------------------------

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
process_every_n_frames = 5  # Procesar cada 5 frames

while True:
    ret, frame = cap.read()

    if not ret:
        break

    shapeRecognition = model(frame, verbose=False) #busqueda de figuras en el frame


    for fig in shapeRecognition:
        for detection in fig.boxes:

            class_id = int(detection.cls[0])
            label = model.names[class_id] #nombre de la figura detectada

            if label in [target["name"] for target in targets]:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Coordenadas (x1, y1) y (x2, y2)
                roi = frame[y1:y2, x1:x2] #area del objeto detectado
                colorName = detectarColor(roi) #detectar el color del objeto
                if colorName in [target["color"] for target in targets]:
                    letter = OCR(roi)
                    if letter in [target["letter"] for target in targets]:
                        print(f"Objeto detectado: {label} de color {colorName} con la letra {letter}")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0 ,0), 2)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                print(f"esta figura es un {label} y no está dentro de los targets")
                                                            
                                    

                        
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
                    


