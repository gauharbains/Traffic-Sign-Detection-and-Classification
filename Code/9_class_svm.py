import cv2 as cv
import numpy as np
import glob
import bisect 
from docutils.parsers import null
from skimage import color
#from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
import pickle

def convert_lab(image):
   clahe = cv.createCLAHE(clipLimit=1., tileGridSize=(1,1))
   lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
   l, a, b = cv.split(lab)  # split on 3 different channels
   l2 = clahe.apply(b)  # apply CLAHE to the L-channel
   lab = cv.merge((l,a,l2))  # merge channels
   img2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # convert from LAB to BGR
   return img2

def get_hog(image):
    image=cv.resize(image,(64,64))
    winSize = (64,64) 
    blockSize = (16,16) 
                        
    blockStride = (8,8) 
    cellSize = (8,8)
    nbins = 16 
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 2
    nlevels = 64
    SignedGradients = True     
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,
                           nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,
                           gammaCorrection,nlevels,SignedGradients)    
    descriptor = hog.compute(image) # Vector of very high dimension   
    return descriptor

t_signs=[45,21,38,35,17,1,14,19,99]
x_train=np.empty((1,3136))
y_train=np.empty((1,1))
for i in t_signs:
    if (i <10):
        path = "TSR/Training/0000%s/*.PPM" %i
    elif (i==99):
        path = "TSR/Training/000%s/*.jpg" %i
        
    else:        
        path = "TSR/Training/000%s/*.PPM" %i
    filenames = [img for img in glob.glob(path)]
    filenames.sort() 
    for j in range(0,len(filenames)):
        image=cv.imread(filenames[j],0)
        descriptor=get_hog(image)
        x_train=np.vstack((x_train,descriptor.T))
        y_train=np.vstack((y_train,i))
x_train=np.delete(x_train,(0),axis=0)
y_train=np.delete(y_train,(0),axis=0)
data_frame=np.hstack((x_train,y_train))
np.random.shuffle(data_frame)
r,c=data_frame.shape
x_train=data_frame[:,:c-1]
y_train=data_frame[:,-1]

x_test=np.empty((1,3136))
y_test=np.empty((1,1))
for i in t_signs:
    if (i <10):
        path = "TSR/Testing/0000%s/*.PPM" %i
    elif (i==99):
        path = "TSR/Testing/000%s/*.jpg" %i
    else:        
        path = "TSR/Testing/000%s/*.PPM" %i
    filenames = [img for img in glob.glob(path)]
    filenames.sort() 
    for j in range(0,len(filenames)):
        image=cv.imread(filenames[j],0)
        descriptor=get_hog(image)
        x_test=np.vstack((x_test,descriptor.T))
        y_test=np.vstack((y_test,i))
x_test=np.delete(x_test,(0),axis=0)
y_test=np.delete(y_test,(0),axis=0)
y_test=y_test.ravel()
clf = svm.SVC(gamma=0.01)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))

img=cv.imread('test2.PPM')
cv.imshow('dd',img)
des=get_hog(img)
print(clf.predict(des.T))

filename = 'finalized_model_9class.sav'
pickle.dump(clf, open(filename, 'wb'))




