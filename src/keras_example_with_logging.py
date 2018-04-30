# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:36:22 2018

@author: Erik
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import get_data as gd
import numpy as np
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
import pandas as pd
import keras
import sklearn
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
N_EXAMPLES = 10000
EPOCHS = 4
BATCH_SIZE = 32 

print('reading data')
patches = gd.get_file_paths(N_EXAMPLES, False)
X_train, X_test, y_train, y_test = gd.get_data(patches, False)
#X_train, y_train = gd.convert_to_dataframe(X_train, y_train) 
#X_test, y_test = gd.convert_to_dataframe(X_test, y_test) 
print("Done reading data")


y_train_hot = to_categorical(y_train)
y_traindf = pd.DataFrame(data = y_train_hot, columns = ['IDC(-1)', 'IDC(+)'])

y_test_hot = to_categorical(y_test)
y_testdf = pd.DataFrame(data = y_test_hot, columns = ['IDC(-1)', 'IDC(+)'])

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


model = Sequential()
model.add(Dense(1024, activation = 'relu', input_dim = 7500))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = keras.optimizers.SGD(lr = 0.001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
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


history = model.fit_generator(datagen.flow(X_train, y_traindf, batch_size=BATCH_SIZE))

score = model.evaluate(X_train, y_traindf)
y_pred = model.predict(X_train)




y_pred = np.argmax(y_pred, axis = 1)
y_act = np.argmax(np.array(y_traindf), axis = 1)
print("Train stats")
print("Accuracy: ", end = '')
print(score[1])
print("F1 score: ", end = '')
print(f1_score(y_act, y_pred))

print()
score = model.evaluate(X_test, y_testdf)
y_pred = model.predict(X_test)

map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
print('\n', sklearn.metrics.classification_report(np.where(y_testdf > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')

y_pred = np.argmax(y_pred, axis = 1)
y_act = np.argmax(np.array(y_testdf), axis = 1)
print("Test stats")
print("Accuracy: ", end = '')
print(score[1])
print("F1 score: ", end = '')
print(f1_score(y_act, y_pred))

        


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



history=model.fit_generator(datagen.flow(X_train, y_traindf, batch_size=BATCH_SIZE),
                    steps_per_epoch=round(len(X_train) / BATCH_SIZE), epochs=EPOCHS,class_weight=class_weight, validation_data = [X_test, y_testdf],callbacks = [MetricsCheckpoint('logs')])


with open('deep_feedforward_train.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in history['acc']:
      file.write("%s\n" % item)
      
with open('deep_feedforward_train_test.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in history['val_acc']:
      file.write("%s\n" % item)
