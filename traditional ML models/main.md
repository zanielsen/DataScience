# Traditional Machine Learning Models - KNN + Random Forest

## Feature Extraction

### Histogram of Oriented Gradients (HOG)
 - Histogram of Oriented Gradients, also known as HOG, is a feature descriptor like the Canny Edge Detector, SIFT (Scale Invariant and Feature Transform) . It is used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in the localized portion of an image. The HOG descriptor focuses on the structure or the shape of an object. It is better than any edge descriptor as it uses magnitude as well as angle of the gradient to compute the features. For the regions of the image it generates histograms using the magnitude and orientations of the gradient. (https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f)

 - KNN and Random Forest did not perform well with features extracted from HOG
    - The models were not able to detect a valid accuracy (0.0) for the test images
    - We can presume the models were not able to learn anything or pick up any patterns from HOG features
    - Additionally, Random Forest runtime is extremely long for 10+ trees

- Overall, HOG is not a good method for extracting meaningful features from images of acne when utilizing traditional ML models KNN and Random Forest

### SIFT

### LBP

### CNN 

## Models

### KNN (K Nearest Neighbors)

### Random Forest

