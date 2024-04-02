# Evaluating YOLO and Efficient DET Models
In an attempt to determine which model is best for the test of acne detection.

## YOLO Models
Tried [YOLOv8n, YOLOv8s, YOLOv8x](https://docs.ultralytics.com/tasks/detect/). YOLOv8n is the fastest of them all, and though the larger YOLO models, YOLOv8s and YOLOv8x have higher mAP50-95 scores, YOLOv8n is only 0.02 down from YOLOv8x and 0.01 from YOLOv8s. So, YOLOv8n, with a confidence of 0.5, comes out on top.

Tried three different confidence levels. Assumed that we'd like more precision/false negative (P) over recall/false positive (R)
because we'd rather be nice to people and say they don't have acne when they do, instead of saying they have acne when they don't.
So, a confidence of 0.25 would maximize overall accuracy of the model, and a confidence of 0.5 would be a "nice" model that would 
rarely tell people they have acne when they don't. 

CONFIDENCE 0.01
| Model | Class  |   Images | Instances   |   P  |        R   |   mAP50 | mAP50-95 |
|-------|--------|----------|------------|-----------|-----------|---------|-----------| 
| yolov8n|  all  |       56 |       643  |    0.666  |    0.573  |    0.627|      0.277|
| yolov8x|  all  |       56 |       643  |    0.599  |    0.591  |    0.589|      0.278| 

CONFIDENCE 0.20
| Model | Class  |   Images | Instances   |   P  |        R   |   mAP50 | mAP50-95 |
|-------|--------|----------|------------|-----------|-----------|---------|-----------| 
| yolov8s|  all  |       56 |       643  |    0.571  |    0.585  |    0.602|      0.294|
| yolov8x|  all  |       56 |       643  |    0.738  |    0.299  |    0.515|      0.285|

CONFIDENCE 0.25
| Model | Class  |   Images | Instances   |   P  |        R   |   mAP50 | mAP50-95 |
|-------|--------|----------|------------|-----------|-----------|---------|-----------| 
| yolov8n |  all |        56|        643 |      0.63 |     0.586 |      0.62|      0.296|
| yolov8s |  all |        56|        643 |     0.571 |     0.585 |     0.595|      0.294|
| yolov8x |  all |        56|        643 |     0.626 |     0.565 |     0.577|        0.3| 

CONFIDENCE 0.5
| Model | Class  |   Images | Instances   |   P  |        R   |   mAP50 | mAP50-95 |
|-------|--------|----------|------------|-----------|-----------|---------|-----------| 
| yolov8n|  all   |      56  |      643   |   0.836   |   0.317   |   0.569 |     0.298 |
| yolov8s|  all   |      56  |      643   |   0.745   |   0.369   |   0.558 |     0.299 |
| yolov8x|  all   |      56  |      643   |   0.738   |   0.299   |   0.515 |     0.285 |

## Efficient DET 
