import cv2
import os
import easyocr
import supervision as sv
from ultralytics import YOLO

model_medicine = YOLO(r"C:\python\numpy\medmodel1.pt")  

reader = easyocr.Reader(['en'])

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")
    exit()

screen_width = 1920  
screen_height = 1080  

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    results_medicine = model_medicine(frame)[0]

    detections_medicine = sv.Detections.from_ultralytics(results_medicine)
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections_medicine)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections_medicine)

    result = reader.readtext(frame)
    
    if result:
        for detection in result:
            bbox, text, confidence = detection

            top_left, top_right, bottom_right, bottom_left = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            cv2.putText(frame, text, (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            print(f"Detected text: {text}, Confidence: {confidence}")

    annotated_image = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_AREA)

    cv2.imshow('Webcam - Medicine Detection with OCR', annotated_image)

    if cv2.waitKey(1) & 0xFF == 27:  
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
