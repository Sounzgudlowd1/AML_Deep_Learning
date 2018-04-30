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
N_EXAMPLES = 10000


print('reading data')
patches = gd.get_file_paths(N_EXAMPLES, False)
X_train, X_test, y_train, y_test = gd.get_data(patches, False)
X_train, y_train = gd.convert_to_dataframe(X_train, y_train) 
X_test, y_test = gd.convert_to_dataframe(X_test, y_test) 
print("Done reading data")


y_train_hot = to_categorical(y_train)
y_traindf = pd.DataFrame(data = y_train_hot, columns = ['IDC(-1)', 'IDC(+)'])

y_test_hot = to_categorical(y_test)
y_testdf = pd.DataFrame(data = y_test_hot, columns = ['IDC(-1)', 'IDC(+)'])


model = Sequential()
model.add(Dense(1024, activation = 'relu', input_dim = 7500))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = keras.optimizers.SGD(lr = 0.001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x = X_train, y = y_traindf, epochs = 20, batch_size = 32)

score = model.evaluate(X_train, y_traindf)
y_pred = model.predict(X_train)


map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}


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
print('\n', sklearn.metrics.classification_report(np.where(y_testdf > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')

y_pred = np.argmax(y_pred, axis = 1)
y_act = np.argmax(np.array(y_testdf), axis = 1)
print("Test stats")
print("Accuracy: ", end = '')
print(score[1])
print("F1 score: ", end = '')
print(f1_score(y_act, y_pred))


