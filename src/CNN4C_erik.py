#C:\Users\skhan\Anaconda3\Lib\site-packages\keras  
import pandas as pd
import numpy as np
import os
import glob
import itertools
import fnmatch
import random
import matplotlib.pylab as plt
import seaborn as sns
import cv2

import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, Conv1D, MaxPool2D, MaxPool1D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from PIL import Image
import numpy as np
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value ########
from skimage import filters  #######
from skimage.exposure import rescale_intensity  ####
#import matplotlib.pyplot as plt                                                                           ########                                                       
#import matplotlib.image as img
from scipy.misc import toimage  ####
#import cv2       
#from pathlib import Path
#import os=
from keras.utils import np_utils
from sklearn.utils import class_weight
import tensorflow as tf
K.set_image_dim_ordering('tf')


imagePatches = glob.glob('../data/**/*.png', recursive=True)
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

from skimage import io
from skimage import color


color.rgb2gray(io.imread(imagePatches[0]))



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

def plot_training_curve(history1,history2,history3,history4):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)

    h1=history1.history['acc']
    h1.insert(0,0)
    plt.plot(h1)
    
    h2=history2.history['acc']
    h2.insert(0,0)
    plt.plot(h2)
    
    h3=history3.history['acc']
    h3.insert(0,0)
    plt.plot(h3)
    
    h4=history4.history['acc']
    h4.insert(0,0)    
    plt.plot(h4)
    
    plt.title('model-train accuracy')
    plt.ylabel('training accuracy')
    plt.xlabel('epoch')
    plt.legend(['Adam', 'SGD','AdaGrad','RMSprop'], loc='lower center', bbox_to_anchor=(0.5, -.15),   ncol=4, fancybox=True, shadow=True )
    plt.savefig('./training_accuracy_curve.png')


def plot_test_curve(history1,history2,history3,history4):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    
    h1=history1.history['val_acc']
    h1.insert(0,0)
    plt.plot(h1)
    
    h2=history2.history['val_acc']
    h2.insert(0,0)
    plt.plot(h2)
    
    h3=history3.history['val_acc']
    h3.insert(0,0)
    plt.plot(h3)

    h4=history4.history['val_acc']
    h4.insert(0,0)    
    plt.plot(h4)
    
    plt.title('model-test accuracy')
    plt.ylabel('testing accuracy')
    plt.xlabel('epoch')
    plt.legend(['Adam', 'SGD','AdaGrad','RMSprop'], loc='lower center', bbox_to_anchor=(0.5,-0.15),   ncol=4, fancybox=True, shadow=True )
    plt.savefig('./testing_accuracy_curve.png')



class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)
        
        
def CNNadam(a,b,c,d,e,f, colorshape,batch_size,epochs):    ############  colorshpae=1 for grayscale   3 for RGB
    """
    Run Keras CNN: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    """
    #batch_size = 128
    num_classes = 2
    #epochs = 8
    img_rows, img_cols = a.shape[1],a.shape[2]
   # img_rows,img_cols=50,50
    input_shape = (img_rows, img_cols, colorshape) 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,strides=e,padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))                  ####OR MAX????????
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, input_dim=img_rows**2,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    
    history=model.fit_generator(datagen.flow(a,b, batch_size=batch_size),
                        steps_per_epoch=round(len(a) / batch_size), epochs=epochs,class_weight=f, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])
    score = model.evaluate(c,d, verbose=0)
    print('\nKeras CNN #1C - accuracy:', score[1],'\n')
    y_pred = model.predict(c)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    
    return(history)
    
    
    
def CNNsgd(a,b,c,d,e,f, colorshape,batch_size,epochs):    ############  colorshpae=1 for grayscale   3 for RGB
    """
    Run Keras CNN: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    """
    #batch_size = 128
    num_classes = 2
    #epochs = 8
    img_rows, img_cols = a.shape[1],a.shape[2]
   # img_rows,img_cols=50,50
    input_shape = (img_rows, img_cols, colorshape) 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,strides=e,padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))                  ####OR MAX????????
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, input_dim=img_rows**2,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    
    history=model.fit_generator(datagen.flow(a,b, batch_size=batch_size),
                        steps_per_epoch=round(len(a) / batch_size), epochs=epochs,class_weight=f, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])
    score = model.evaluate(c,d, verbose=0)
    print('\nKeras CNN #1C - accuracy:', score[1],'\n')
    y_pred = model.predict(c)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    
    return(history)
    


def CNNadagrad(a,b,c,d,e,f, colorshape,batch_size,epochs):    ############  colorshpae=1 for grayscale   3 for RGB
    """
    Run Keras CNN: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    """
    #batch_size = 128
    num_classes = 2
    #epochs = 8
    img_rows, img_cols = a.shape[1],a.shape[2]
   # img_rows,img_cols=50,50
    input_shape = (img_rows, img_cols, colorshape) 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,strides=e,padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))                  ####OR MAX????????
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, input_dim=img_rows**2,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adagrad(),
                  metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    
    history=model.fit_generator(datagen.flow(a,b, batch_size=batch_size),
                        steps_per_epoch=round(len(a) / batch_size), epochs=epochs,class_weight=f, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])
    score = model.evaluate(c,d, verbose=0)
    print('\nKeras CNN #1C - accuracy:', score[1],'\n')
    y_pred = model.predict(c)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    
    return(history)
    

    

def CNNrms(a,b,c,d,e,f, colorshape,batch_size,epochs):    ############  colorshpae=1 for grayscale   3 for RGB
    """
    Run Keras CNN: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
    """
    #batch_size = 128
    num_classes = 2
    #epochs = 8
    img_rows, img_cols = a.shape[1],a.shape[2]
   # img_rows,img_cols=50,50
    input_shape = (img_rows, img_cols, colorshape) 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,strides=e,padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))                  ####OR MAX????????
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, input_dim=img_rows**2,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    
    print(a.shape)
    print(type(a))
    print(b.shape)
    print(type(b))
    history=model.fit_generator(datagen.flow(a,b, batch_size=batch_size),
                        steps_per_epoch=round(len(a) / batch_size), epochs=epochs,class_weight=f, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])
    score = model.evaluate(c,d, verbose=0)
    print('\nKeras CNN #1C - accuracy:', score[1],'\n')
    y_pred = model.predict(c)
    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    
    return(history)

X,Y = proc_col(50,50,'none',0,10000)
df = pd.DataFrame()
df["images"]=X
df["labels"]=Y
X2=df["images"]
Y2=df["labels"]
X2=np.array(X2)
imgs0=[]
imgs1=[]
imgs0 = X2[Y2==0] # (0 = no IDC, 1 = IDC)
imgs1 = X2[Y2==1] 
X=np.array(X)
X=X/255.0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
Y_trainHot = np_utils.to_categorical(Y_train, num_classes = 2)
Y_testHot =np_utils.to_categorical(Y_test, num_classes = 2)




class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
print("Old Class Weights: ",class_weight)
    
h1=CNNadam(X_train, Y_trainHot, X_test, Y_testHot,10,class_weight ,3,32,50)
with open('adament_train.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in h1.history['acc']:
      file.write("%s\n" % item)
with open('adament_test.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in h1.history['val_acc']:
      file.write("%s\n" % item)


 


h2=CNNsgd(X_train, Y_trainHot, X_test, Y_testHot,10,class_weight ,3,32,50)
with open('sgdent_train.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in h2.history['acc']:
      file.write("%s\n" % item)
      
with open('sgdent_test.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in h2.history['val_acc']:
      file.write("%s\n" % item)
      
      
h3=CNNadagrad(X_train, Y_trainHot, X_test, Y_testHot,10,class_weight ,3,32,50)
with open('adaent_train.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in h3.history['acc']:
      file.write("%s\n" % item)
      
with open('adaent_test.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in h3.history['val_acc']:
      file.write("%s\n" % item)
 
 
h4=CNNrms(X_train, Y_trainHot, X_test, Y_testHot,10,class_weight ,3,32,50)
with open('rmsent_train.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in h4.history['acc']:
      file.write("%s\n" % item)
with open('rmsent_test.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in h4.history['val_acc']:
      file.write("%s\n" % item)


#expandedCNNAugment(X_train, Y_trainHot, X_test, Y_testHot,10,class_weight ,3,32,8)  
######e=1 is bad  naive fscore and llong run time.   e=3  changes little    e=10 has a some better fsores
# Changing kernel size changes very little. 
#Chaginge averagepooling vs maxpooling changes little
### input=row^2   resulted in lower F-score for +
      
print(h1.history['acc'])     
print(h2.history['acc'])     
print(h3.history['acc'])     
print(h4.history['acc'])      
          
a=[]   
with open('adament_train.txt') as f:
    for line in f:
        v=float(line)
        a.append(v)
        #np.vstack((n,line))

a2=[]
with open('adament_test.txt') as f:
    for line in f:
        v=float(line)
        a2.append(v)
        #np.vstack((n,line))
        
        
s=[]   
with open('sgdent_train.txt') as f:
    for line in f:
        v=float(line)
        s.append(v)
        #np.vstack((n,line))

s2=[]
with open('sgdent_test.txt') as f:
    for line in f:
        v=float(line)
        s2.append(v)
        #np.vstack((n,line))


ada=[]   
with open('adaent_train.txt') as f:
    for line in f:
        v=float(line)
        ada.append(v)
        #np.vstack((n,line))

ada2=[]
with open('adaent_test.txt') as f:
    for line in f:
        v=float(line)
        ada2.append(v)
        #np.vstack((n,line))
        
rms=[]   
with open('rmsent_train.txt') as f:
    for line in f:
        v=float(line)
        rms.append(v)
        #np.vstack((n,line))

rms2=[]
with open('rmsent_test.txt') as f:
    for line in f:
        v=float(line)
        rms2.append(v)
        #np.vstack((n,line))
  
    
    
plt.figure(figsize=(30,15))
plt.subplot(1,2,1)
plt.plot(a)
plt.plot(s)
plt.plot(ada)
plt.plot(rms)

plt.title('model-train accuracy')
plt.ylabel('training accuracy')
plt.xlabel('epoch')
plt.legend(['Adam', 'SGD','AdaGrad','RMSprop'], loc='lower center', bbox_to_anchor=(0.5, -.15),   ncol=4, fancybox=True, shadow=True )
plt.savefig('./ENT training_accuracy_curve.png')



plt.figure(figsize=(30,15))
plt.subplot(1,2,1)
plt.plot(a2)
plt.plot(s2)
plt.plot(ada2)
plt.plot(rms2)

plt.title('model-test accuracy')
plt.ylabel('testing accuracy')
plt.xlabel('epoch')
plt.legend(['Adam', 'SGD','AdaGrad','RMSprop'], loc='lower center', bbox_to_anchor=(0.5, -.15),   ncol=4, fancybox=True, shadow=True )
plt.savefig('./ENT testing_accuracy_curve.png')