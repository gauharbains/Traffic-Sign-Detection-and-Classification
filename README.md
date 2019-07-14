# Traffic-Sign-Detection-and-Classification
## Output 
Sample outputs :


![output1](https://user-images.githubusercontent.com/48079888/61189512-6a2c4900-a65c-11e9-967f-edc2da90c8d8.gif)
![output2](https://user-images.githubusercontent.com/48079888/61189513-6bf60c80-a65c-11e9-966d-362897f2d205.gif)



## Pipeline

### 1. Outline 

![pipeline1](https://user-images.githubusercontent.com/48079888/61189544-e888eb00-a65c-11e9-96c7-f20e9ffe6811.JPG)

### 2. Steps

#### 2.1 Image Preprocessing 
The first step was to denoise the image. We used the OpenCV function cv2.fastNlMeansDenoisingColored
to remove the noise from the image. In this technique, we identify the noise pixel, and take a small window
around it and search for similar window. We take the average of those similar windows and replace the
pixel value with the average.

#### 2.2 Contrast Stretching 
The next step was to use the technique of contrast stretching to improve the contrast in the image. To
implement this, we first created the histogram for each channel. From that histogram, we created a
cumulative histogram. We have passed the lower and higher threshold value of 15% and 85%.
Corresponding to that percent value in the cumulative histogram, we have cut the histogram at
corresponding value and stretched it from 0 to 255. We have added the stretched histogram with the
original histogram to improve the contract in the image. This was done for all the three-color channels.

#### 2.3 Intensity Normalization & MSER
The next step was to apply MSER to the images that have been converted to the grey scale. MSER was
generated with minimum area set to 1000, and maximum area set to 4000 using . Of the extracted regions,
a bounding box was drawn to highlight the region that was extracted.

![Capture](https://user-images.githubusercontent.com/48079888/61189620-099e0b80-a65e-11e9-9b6f-6a3c7f7c6212.JPG)

Output from MSER

#### 2.4 HOG Feature
Of the region that was highlighted, the HOGDescriptor from the OpenCV library was used to get the HOG
descriptors for the specific regions. The HOG descriptor is stored and then later used for classification in
SVM (Singular Vector Machine). MSER was detecting multiple regions that do not contains the signs. Thus,
we have used two SVMs, one to detect and eliminate those regions without the signs and the second one
that classifies the signs.

#### 2.5 Classification
We have used two SVM for the process of classification. The first SVM is used to classify if the region that
has been highlighted is a sign or not and this SVM is binary in nature. It only has two outputs. Is sign and
not sign. The accuracy of this SVM is about 96%.Those regions that are been classified as sign are passed through another SVM. This SVM is used to classify the signs into 8 signs, and the 9 output for this SVM is no sign. The 9th output of this SVM is no sign and is
set so that any regions with no signs that were missed by the previous SVM, are covered by this one.
Based on the output of the second SVM, the signs are classified as either of the 8 signs. The sign sample
is pasted along side the detected sign.

#### 2.6 Sample images used for training the svm are shown below: 

![00035_00001](https://user-images.githubusercontent.com/48079888/61189699-153e0200-a65f-11e9-9170-7144ea7992c4.jpg) ![00023_00000](https://user-images.githubusercontent.com/48079888/61189700-153e0200-a65f-11e9-86a2-0ebd2cc3e83b.jpg)

![00406_00001](https://user-images.githubusercontent.com/48079888/61189729-75cd3f00-a65f-11e9-8867-88921e54a4ab.jpg) ![00904_00002](https://user-images.githubusercontent.com/48079888/61189730-75cd3f00-a65f-11e9-95a0-b1e79475720f.jpg)

