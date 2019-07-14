import cv2 as cv
import numpy as np
import glob
import bisect 
from docutils.parsers import null
from skimage import color
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
import pickle

#
def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    tol = tol/2
    src = src.astype(float)
    assert len(src.shape) == 2 ,'Input image should be 2-dims'
    tol = max(0, min(100, tol))

    if tol > 0:

        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]
        cum = hist.copy()
        cum = np.cumsum(cum)
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[vs<0]=0
    vs = (vs*scale) + vout[0]
    vs[vs>vout[1]] = vout[1]
    dst = vs.astype(np.uint8)
    return dst

#contrast stretching: local contrast normal
def clahecontrast(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return_image = clahe.apply(image)
    return return_image

       
def contrastnormalization(image):
    image_blue = image[:,:,0]
    image_green = image[:,:,1]
    image_red = image[:,:,2]
    arr_blue = np.asarray(image_blue)
    arr_green = np.asarray(image_green)
    arr_red = np.asarray(image_red)
    
    arr_blue= clahecontrast(arr_blue)
    arr_green= clahecontrast(arr_green)
    arr_red= clahecontrast(arr_red)
    
    cn_blue=imadjust(arr_blue,60, [arr_blue.min(),arr_blue.max()],(arr_blue.min(),arr_blue.max()))
    cn_green=imadjust(arr_green,60, [arr_green.min(),arr_green.max()], (arr_green.min(),arr_green.max()))
    cn_red=imadjust(arr_red,60, [arr_red.min(),arr_red.max()], (arr_red.min(),arr_red.max()))
    cn = np.dstack([cn_blue,cn_green,cn_red])
    
    return cn

def intensitynormalization(image):
    image = image.astype(float)
    image_sum = np.sum(image,axis =2)
    minimum_red = np.minimum((image[:,:,2]-image[:,:,0]),(image[:,:,2]-image[:,:,1]))/image_sum
    C_red = np.maximum(0,minimum_red)
    C_red_max = np.amax(C_red)
    C_red = (C_red*255)/C_red_max
    C_red = C_red.astype(np.uint8)
    
    minimum_blue = (image[:,:,0]-image[:,:,2])/image_sum
    C_blue = np.maximum(0,minimum_blue)
    C_blue_max = np.amax(C_blue)
    C_blue = (C_blue*255)/C_blue_max
    C_blue = C_blue.astype(np.uint8)
    arr_blue = clahecontrast(C_blue)
    arr_red = clahecontrast(C_red)
    arr = np.maximum(arr_blue,arr_red)
    return arr

def filter_intrestregion(x,y,w,h):
    check = False
    if (h/w > 0.6 and h/w <1.3 and w*h > 3456 and w*h <89856):
       check = True     
    return check

def MSER(img,img1):
    first = True
    vis =img.copy()
    mser = cv.MSER_create()
    mser.setMinArea(1000)
    mser.setMaxArea(4000)
    coordinate_array=np.zeros((1,4))
    regions = mser.detectRegions(img)
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    for i, contour in enumerate(hulls):
        x,y,w,h = cv.boundingRect(contour)
        if filter_intrestregion(x,y,w,h):
            if first == False :
                coordinate_array = np.vstack((coordinate_array,np.array([x,y,w,h]))) 
            if first == True :
                coordinate_array = np.array([x,y,w,h])
                first = False 

    return vis, coordinate_array
    
###########################################################################################
def get_hog(image):
    image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
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

def extract_roi(img,rec_cd):
    """ rec_cd is a list or tuple of 4 elements (x,y,w,h)"""
    x,y,w,h=rec_cd    
    roi_image=img[y:y+h,x:x+w]    
    return roi_image

filename = 'finalized_model_9class.sav'
clf_traffic_clasifier = pickle.load(open(filename,'rb'))

filename = 'negative_positive_svm.sav'
clf_binary_classiier = pickle.load(open(filename,'rb'))

def TrafficSignRecognition(img):

    image1 = contrastnormalization(img)
    image_arr = intensitynormalization(image1)
    image_detected,coordinate_array = MSER(image_arr,img)
    if(np.sum(coordinate_array)>0):
        for i in coordinate_array:
            roi=extract_roi(img,i)
            x,y,w,h=i[0],i[1],i[2],i[3]            
            des=get_hog(roi)
 
            if (((clf_binary_classiier.predict(des.T))[0])==1):
                traffic_sign_number=int(clf_traffic_clasifier.predict(des.T)[0])
                if traffic_sign_number==99:
                    continue
                img[y:y+64,x-64:x]=cv.imread("clean_signs/%s.PNG"%traffic_sign_number)
                cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    cv.imwrite("output1/out%s.jpg"%count,img)

path = "TSR/denoised_images/*.jpg"
filenames = [img for img in glob.glob(path)]
filenames.sort() 
count = 0
for i in range (len(filenames)):
#    print(i)
    count +=1
    write_name = 'input/image'+format(count)+'.jpg'
    image=cv.imread(write_name)
#    image=cv.imread(filenames[i])
    TrafficSignRecognition(image) 
    
    
cv.destroyAllWindows()
#print(count)

