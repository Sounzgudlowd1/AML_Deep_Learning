# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:15:44 2018

@author: skhan
"""

#C:\Users\skhan\Anaconda3\Lib\site-packages\keras  
import numpy as np
import glob
import fnmatch
import cv2
from PIL import Image
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value ########
from skimage import filters  #######
from skimage.exposure import rescale_intensity  ####
from scipy.misc import toimage  ####

imagePatches = glob.glob('C:/Users/skhan/Desktop/Main/School/UIC/Academics/Semester_2/Adv_ML/Assignments/DLProject/DLP/data/**/*.png', recursive=True)
patternZero = '*class0.png'
patternOne = '*class1.png'
classZero = fnmatch.filter(imagePatches, patternZero)
classOne = fnmatch.filter(imagePatches, patternOne)
print("IDC(-)\n\n",classZero[0:5],'\n')
print("IDC(+)\n\n",classOne[0:5])

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)
@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)






###This preprocessing function converts the RGB to grayscale.  The "coloring" dictates which, if any, filtering should be applied before conversion.  Filterting seems to improve grayscale 
##### It also resizes based on the inputs
def proc_grey(WIDTH,HEIGHT,coloring, start,end):    ##start , end indicate which images we want to use
    x = []
    y = []
    i=start;
    for imge in imagePatches[start:end]:
        i=i+1
        if i%500==0:
            print("Image # ", i, " is bein preprocessed");
        image=Image.open(imge)
        if coloring=='hsv':
            image_out=toimage(rescale_intensity(1 - sobel_hsv(np.asarray(image)))).convert('L')
        elif coloring=='each':
            image_out=toimage(rescale_intensity(1 - sobel_each(np.asarray(image)))).convert('L')
        else:
            image_out=image.convert('L') ####LA
            
        image_out = np.asarray(image_out.resize((WIDTH,HEIGHT)))
        image_out=np.reshape(image_out,(WIDTH,HEIGHT,1))
        x.append(image_out)
        if imge in classZero:
            y.append(0)
        elif imge in classOne:
            y.append(1)
    
    return x, y

###This preprocessing function filters the RGB. The "coloring" dictates which, if any, filtering should be applied. This may not be very useful, but can still be used for resizing the original photos
##### It also resizes based on the inputs


def proc_col(WIDTH,HEIGHT,coloring,start,end):    ##start , end indicate which images we want to use
    x = []
    y = []
    i=start
    for imge in imagePatches[start:end]:    
        i=i+1
        if i%500==0:
            print("Image # ", i, " is bein preprocessed");

        image=Image.open(imge)
        if coloring=='hsv':
            image_out=toimage(rescale_intensity(1 - sobel_hsv(np.asarray(image))))
        elif coloring=='each':
            image_out=toimage(rescale_intensity(1 - sobel_each(np.asarray(image))))
        else:
            image_out=image
        
        image_out = np.asarray(image_out.resize((WIDTH,HEIGHT)))
        x.append(image_out)
        if imge in classZero:
            y.append(0)
        elif imge in classOne:
            y.append(1)

    return x, y


def proc_images(lowerIndex,upperIndex):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """ 
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    i=lowerIndex;
    for img in imagePatches[lowerIndex:upperIndex]:
        i=i+1;
        if i%500==0:
            print("Image # ", i, " is bein preprocessed");
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x,y

