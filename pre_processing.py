from PIL import Image
import numpy as np
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt                                                                    
import glob                                                                
import matplotlib.image as img
from scipy.misc import toimage
import cv2       
from pathlib import Path
import os
import fnmatch




###glob searches for all files  end with ".png"  Pathway will havee to chang depending on where the files are stored
imagePatches = glob.glob('C:/Users/skhan/Desktop/Main/School/UIC/Academics/Semester_2/Adv_ML/Assignments/DLProject/DLP/data/**/*.png', recursive=True)


#group the photos based on weather they belong class 1 or class 0
patternZero = '*class0.png'
patternOne = '*class1.png'
classZero = fnmatch.filter(imagePatches, patternZero)
classOne = fnmatch.filter(imagePatches, patternOne)



#################filter functions to change the photos
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
    i=0;
    for imge in imagePatches[start:end]:
        i=i+1
        print('Image #',i," being preprocessed")
        image=Image.open(imge)
        if coloring=='hsv':
            image_out=toimage(rescale_intensity(1 - sobel_hsv(np.asarray(image)))).convert('L')
        elif coloring=='each':
            image_out=toimage(rescale_intensity(1 - sobel_each(np.asarray(image)))).convert('L')
        else:
            image_out=image.convert('L') ####LA
            
        image_out = np.asarray(image_out.resize((WIDTH,HEIGHT)))
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

    for imge in imagePatches[0:10]:        
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


###Intensity scaling of the pixel matrices should aslo be done if color preprocessing is used
        ###X=np.array(x)
        ###X=X/255.0             
