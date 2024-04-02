from ultralytics import YOLO
import cv2
import cvzone
import math

# Code to train model
'''
model = YOLO('yolov8n.pt') # load a pre-trained model 
results = model.train(data='data.yaml', epochs=200, imgsz=640)
'''

model = YOLO('yolov8n-acne.pt')  # load model trained on acne training data (yolov8n model)

model.val(data="./data.yaml", conf=0.5)

# Show an example of model's performance on test data
input_image = cv2.imread("./test/images/acne-239_jpg.rf.08487e45a5ca975c038eabe8464b0c87.jpg")
results = model.predict(source=input_image, conf=0.2)

# Detection boxes
print(results[0].boxes.xyxy)

# Output an image (output.jpg) with detection boxes drawn
classNames = ["Acne"]
'''
results = model(input_image, stream=True)
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cvzone.cornerRect(input_image, (x1, y1, w, h))

        conf = math.ceil((box.conf[0]*100))/100

        cls = box.cls[0]
        name = classNames[int(cls)]

        cvzone.putTextRect(input_image, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5)
cv2.imwrite('output.jpg', input_image)
'''
