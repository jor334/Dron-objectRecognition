# Dron-objectRecognition
Reconocimiento de objetos para proyecto de Dron/Robocol


## Division de archivos

models:
- best.pt es un custom model basado en yolov8 entrenado con distintas figuras geometricas de un dataset
- yolov8m, yolov8s, yolov10m: modelos preentrenados de ultralytics

objectRecognition:
- oldODCL: sistema de deteccion de objetos en tiempo real usando los requisitos de de suas2024, detectando figuras, colores  y letras
- ODCL: sistema actual de deteccion en tiempo real usando modelo preentrenado de yolov8m
- imgDetection: sistema de deteccion usando yolov8m para imagenes de ejemplo

- dataExamples: imagenes para pruebas de deteccion

