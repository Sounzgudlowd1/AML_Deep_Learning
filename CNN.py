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
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import keras
import sklearn
'''
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
'''


###glob searches for all files  end with ".png"  Pathway will havee to chang depending on where the files are stored

print('Reading patches')
imagePatches = glob.glob('C:/Users/skhan/Desktop/Main/School/UIC/Academics/Semester_2/Adv_ML/Assignments/DLProject/DLP/data/**/*.png', recursive=True)

print();
print('Seperating the classes')
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
    i=start;
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
    i=start
    for imge in imagePatches[start:end]:    
        i=i+1
        print('Image #',i," being preprocessed")

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


'''  One of the original preprocessing functions  from which others are partially based on 
# https://www.kaggle.com/paultimothymooney/predict-idc-in-breast-cancer-histology-images/notebook
def proc_images(lowerIndex,upperIndex):

    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    for imge in imagePatches[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(imge)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        
        if imge in classZero:
            y.append(0)
        elif imge in classOne:
            y.append(1)
        else:
            return
            
    return x, y
'''



def runCNN(a,b,c,d,e,f,batch_size,epochs):
    """
    Run Keras CNN: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    """
    #batch_size=32
    #epochs = 8
    num_classes = 2
    img_rows, img_cols = a.shape[1],a.shape[2]
 #  img_rows,img_cols=50,50
    input_shape = (img_rows, img_cols, 3)
    

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,strides=e))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
   # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False #,  # divide each input by its std
       # zca_whitening=False  # apply ZCA whitening
                                )  
    print("----error here----")
    model.fit_generator(
                        datagen.flow(a,b, batch_size=batch_size),
                        steps_per_epoch=len(a) / 32,
                        epochs=epochs,
                        class_weight=f, 
                        validation_data = [c, d]#,
                        #callbacks = [MetricsCheckpoint('logs')]
                        )
    print("----error here----")
    
    score = model.evaluate(c,d, verbose=0)
    print('\nKeras CNN #1C - accuracy:', score[1],'\n')
    y_pred = model.predict(c)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    






X,Y = proc_col(40,40,'hsv',0,10000)
X=np.array(X)
print();
print('Splitting the test & train data')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

Y_train = np_utils.to_categorical(Y_train, num_classes = 2)
Y_test = np_utils.to_categorical(Y_test, num_classes = 2)


                    ###This section is relevant only if we want to resample to account for minority class
                    '''
                    X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
                    X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]
                    X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
                    X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)
                    #print("X_train Shape: ",X_train.shape)
                    #print("X_test Shape: ",X_test.shape)
                    #print("X_trainFlat Shape: ",X_trainFlat.shape)
                    #print("X_testFlat Shape: ",X_testFlat.shape)
                    
                    
                    #ros = RandomOverSampler(ratio='auto')
                    ros = RandomUnderSampler(ratio='auto')
                    X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
                    X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)
                    
                    # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
                    Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 2)
                    Y_testRosHot = to_categorical(Y_testRos, num_classes = 2)
                    #print("X_train: ", X_train.shape)
                    #print("X_trainFlat: ", X_trainFlat.shape)
                    #print("X_trainRos Shape: ",X_trainRos.shape)
                    #print("X_testRos Shape: ",X_testRos.shape)
                    #print("Y_trainRosHot Shape: ",Y_trainRosHot.shape)
                    #print("Y_testRosHot Shape: ",Y_testRosHot.shape)
                    
                    for i in range(len(X_trainRos)):
                        height, width, channels = 50,50,3
                        X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)
                    #print("X_trainRos Shape: ",X_trainRos.shape)
                    #print("X_trainRosReshaped Shape: ",X_trainRosReshaped.shape)
                    
                    for i in range(len(X_testRos)):
                        height, width, channels = 50,50,3
                        X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)
                    '''

print('Now running the CNN')
runCNN(X_train, Y_train, X_test, Y_test,2,None,128,10)
  
#runCNN(X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot,2,class_weight2)
###Intensity scaling of the pixel matrices should aslo be done if color preprocessing is used
        ###X=np.array(x)
        ###X=X/255.0             