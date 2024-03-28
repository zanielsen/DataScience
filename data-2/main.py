from ultralytics import YOLO
import cv2
import cvzone
import math

# Code to train model
'''
model = YOLO('yolov8n.pt') # load a pre-trained model 
results = model.train(data='data.yaml', epochs=200, imgsz=640)
'''

model = YOLO('best.pt')  # load model trained on acne training data (yolov8n model)

'''
Tried three different confidence levels. Assumed that we'd like more precision/false negative (P) over recall/false positive (R)
because we'd rather be nice to people and say they don't have acne when they do, instead of saying they have acne when they don't.
So, a confidence of 0.25 would maximize overall accuracy of the model, and a confidence of 0.5 would be a "nice" model that would 
rarely tell people they have acne when they don't. 

CONFIDENCE 0.01                 
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 
  all         56        643      0.666      0.573      0.627      0.277
-------------------------------------------------------------------------

CONFIDENCE 0.25
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
  all         56        643       0.63      0.586       0.62      0.296
-------------------------------------------------------------------------

CONFIDENCE 0.5
Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
  all         56        643      0.836      0.317      0.569      0.298
'''

model.val(data="./data.yaml", conf=0.25)

# Show an example of model's performance on test data
input_image = cv2.imread("./test/images/acne-239_jpg.rf.08487e45a5ca975c038eabe8464b0c87.jpg")
results = model.predict(source=input_image, conf=0.25)

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
