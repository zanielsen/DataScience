from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skimage.feature import hog
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import numpy as np
from skimage import color, feature

### Load Images

def load_images(foldername):
    list = []
    for filename in os.listdir(foldername):
        if filename.lower().endswith(".jpg"): 
            image = Image.open(os.path.join(foldername,filename))
            if image is not None:
                list.append(image)
        else:
            label = os.path.join(foldername,filename)
            if label is not None:
                list.append(label)
    return list

x_train = load_images('/Users/hannahzhang/Desktop/DataScience/DataScience/data-2/train/images')
x_test = load_images('/Users/hannahzhang/Desktop/DataScience/DataScience/data-2/test/images')
y_train = load_images('/Users/hannahzhang/Desktop/DataScience/DataScience/data-2/train/labels')
y_test = load_images('/Users/hannahzhang/Desktop/DataScience/DataScience/data-2/test/labels')



### Feature Extraction 

# HOG

def hog_features(list):
    hog_list = []
    for im in list:
        im_resized = im.resize((300, 300))
        im_array = np.array(im_resized)
        im_gray = color.rgb2gray(im_array)
        
        # Compute HOG features
        fd = feature.hog(im_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_list.append(fd)

    return hog_list

hog_x_train = hog_features(x_train)
hog_x_test = hog_features(x_test)

# SIFT

# LBP

# CNN

### K Nearest Neighbors

# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(hog_x_train, y_train)

# y_pred = knn.predict(hog_x_test)
# print(classification_report(y_test, y_pred))

### Random Forest

rf = RandomForestClassifier(n_estimators=40, random_state=42)
rf.fit(hog_x_train, y_train)

y_pred = rf.predict(hog_x_test)

print(classification_report(y_test, y_pred))