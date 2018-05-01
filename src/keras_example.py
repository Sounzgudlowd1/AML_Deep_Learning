# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 13:36:22 2018

@author: Erik
"""

from keras.models import Sequential
from keras.layers import Dense
import get_data as gd
import numpy as np
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
import pandas as pd
import keras
import sklearn


N_EXAMPLES = 10000
BATCH_SIZE = 32
EPOCHS = 50
SOLVER = 'SGD'

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


if SOLVER == 'SGD':
    solver = keras.optimizers.SGD(lr = 0.001)
elif SOLVER == 'Adam':
    solver = keras.optimizers.Adam(lr = 0.0001)
elif SOLVER == 'AdaGrad':
    solver = keras.optimizers.Adagrad(lr = 0.0001)
elif SOLVER == 'RMSProp':
    solver = keras.optimizers.RMSprop(lr = 0.0001)
else:
    print("UNKNOWN OPTIMIZER, HAULTING")

model = Sequential()
model.add(Dense(1024, activation = 'relu', input_dim = 7500))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = solver,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(x = X_train, y = y_traindf, epochs = EPOCHS, batch_size = BATCH_SIZE, 
          validation_data = [X_test, y_testdf])

score = model.evaluate(X_train, y_traindf)
y_pred = model.predict(X_train)





y_pred = np.argmax(y_pred, axis = 1)
y_act = np.argmax(np.array(y_traindf), axis = 1)
print("Train stats")
print("Accuracy: ")
print(score[1])
print("F1 score: ")
print(f1_score(y_act, y_pred))

print()
score = model.evaluate(X_test, y_testdf)
y_pred = model.predict(X_test)


map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
print('\n', sklearn.metrics.classification_report(np.where(y_testdf > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')

y_pred = np.argmax(y_pred, axis = 1)
y_act = np.argmax(np.array(y_testdf), axis = 1)
print("Test stats")
print("Accuracy: ")
print(score[1])
print("F1 score: ")
print(f1_score(y_act, y_pred))


with open('ffnn_train_' + SOLVER + '.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in history.history['acc']:
      file.write("%s\n" % item)
with open('ffnn_test_' + SOLVER + '.txt', 'w') as file:
    file.write("%s\n" % 0.0)
    for item in history.history['val_acc']:
      file.write("%s\n" % item)