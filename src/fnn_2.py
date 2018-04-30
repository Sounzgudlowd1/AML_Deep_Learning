# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 22:51:17 2018

@author: Erik
"""
import tensorflow as tf
import get_data as gd
import numpy as np

#randomly  get some file paths,
from tensorflow.contrib.learn.python import SKCompat
from sklearn.metrics import f1_score
from keras.preprocessing.image import ImageDataGenerator


STEPS = 30000
HIDDEN_UNITS = [2058, 256]
N_EXAMPLES =10000



print("Starting reading data")

#take those file paths, and convert to actual examples as lists of lists, do not randomly split them

patches = gd.get_file_paths(N_EXAMPLES, False)
X_train, X_test, y_train, y_test = gd.get_data(patches, False)
X_train = np.array(X_train, dtype = np.float32)/255.0
y_train = np.array(y_train, dtype = np.bool)
X_test = np.array(X_test, dtype = np.float32)/255.0
y_test = np.array(y_test, dtype = np.bool)
print("Done reading data")

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(X_train[0]))]

classifier = SKCompat(tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                            hidden_units = HIDDEN_UNITS,
                                            optimizer = tf.train.AdamOptimizer(1e-6),
                                            n_classes = 2))

print("Starting fit")
classifier.fit(x = X_train, y = y_train, steps = STEPS)
print("done fitting")

#get predictions
output = classifier.predict(X_test)
np.savetxt('predictions.csv', output['classes'], delimiter = ',')

print("\n\nTEST")
print("F1 score: ", end = '')
print(f1_score(y_test, output['classes']))
print("Accuracy: ", end = '')
print(np.sum(y_test == output['classes'])/len(y_test))
print("True y percent: ", end = '')
print(np.sum(y_test)/len(y_test))
print("Predicted y percent: ", end = '')
print(np.sum(output['classes'])/len(y_train))
print("Total examples: ", end = '')
print(len(y_test))
print("Number of errors: ", end = '')
print(np.sum(y_test != output['classes']))

output = classifier.predict(X_train)
print("\n\nTRAIN")
print("F1 score: ", end = '')
print(f1_score(y_train, output['classes']))
print("Accuracy: ", end = '')
print(np.sum(y_train == output['classes'])/len(y_train))
print("True y percent: ", end = '')
print(np.sum(y_train)/len(y_train))
print("Predicted y percent: ", end = '')
print(np.sum(output['classes'])/len(y_train))
print("Total examples: ", end = '')
print(len(y_train))
print("Number of errors: ", end = '')
print(np.sum(y_train != output['classes']))




