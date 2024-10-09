import cv2
import numpy as np

# Definir un diccionario de colores en formato BGR
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

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se puede acceder a la cámara.")
    exit()

while True:
    # Leer el frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("No se puede recibir el frame (el stream ha terminado?). Saliendo ...")
        break

    # Redimensionar la imagen para agilitar el procesamiento (opcional)
    frame = cv2.resize(frame, (640, 480))

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

    # Crear una imagen con el color predominante
    color_display = np.zeros((200, 200, 3), dtype=np.uint8)
    color_display[:] = predominant_color

    # Agregar el nombre del color en la imagen
    cv2.putText(color_display, color_name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Dibujar un rectángulo en la región central en el frame original
    start_point = (int(width/4), int(height/4))
    end_point = (int(3*width/4), int(3*height/4))
    color_rectangle = (0, 255, 0)  # Verde para el rectángulo
    thickness = 2  # Grosor del rectángulo
    cv2.rectangle(frame, start_point, end_point, color_rectangle, thickness)

    # Mostrar el frame original y el color predominante
    cv2.imshow('Frame Original', frame)
    cv2.imshow('Color Predominante', color_display)

    # Salir si se presiona 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
