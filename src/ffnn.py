# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:19:49 2018

@author: Erik
"""

import tensorflow as tf
import get_data as gd
import numpy as np

HIDDEN_UNITS = [4096]
STEPS = 1
N_EXAMPLES = 10000

'''
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
'''
def train_input_fn():
    pass

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

#randomly  get some file paths,
print("Starting reading data")
patches = gd.get_file_paths(N_EXAMPLES, False)
#take those file paths, and convert to actual examples as lists of lists, do not randomly split them
X_train, X_test, y_train, y_test = gd.get_data(patches, False)

#now just convert to pandas dataframes so that it is easier to train.
X_train, y_train = gd.convert_to_dataframe(X_train, y_train)
X_test, y_test = gd.convert_to_dataframe(X_test, y_test)


print("done loading data")
print("%Positive in test labels: ", end = '')
print(np.sum(y_test)/len(y_test) *100)


#classifier = tf.estimator.DNNClassifier(feature_columns = feature_cols, hidden_units = [10, 10], n_classes = 4)
feature_cols = []
for col in X_train.keys():
    feature_cols.append(tf.feature_column.numeric_column(key=col))
    

classifier = tf.estimator.DNNClassifier(feature_columns = feature_cols, hidden_units = HIDDEN_UNITS, n_classes = 2)
print("built classifier")

classifier.train(input_fn = lambda:train_input_fn(X_train, y_train, 100)
                , steps = STEPS)

test_accuracy = classifier.evaluate(
        input_fn=lambda:eval_input_fn(X_test, y_test, 100))
print("\nTest accuracy: {accuracy:0.3f}\n".format(**test_accuracy))


train_accuracy = classifier.evaluate(
        input_fn=lambda:eval_input_fn(X_train, y_train, 100))

print('\nTrain set accuracy: {accuracy:0.3f}\n'.format(**train_accuracy))

print("Hidden units: ", end = '')
print(HIDDEN_UNITS)
print("Steps: ", end = '')
print(STEPS)
print("N_EXAMPES: ", end = '')
print(N_EXAMPLES)

