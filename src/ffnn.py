# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:19:49 2018

@author: Erik
"""

import tensorflow as tf
import get_data as gd

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()

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


patches = gd.get_file_paths(10000, True)

X_train, X_test, y_train, y_test = gd.get_data(patches, False)

X_train, y_train = gd.convert_to_dataframe(X_train, y_train)
X_test, y_test = gd.convert_to_dataframe(X_test, y_test)


#classifier = tf.estimator.DNNClassifier(feature_columns = feature_cols, hidden_units = [10, 10], n_classes = 4)
feature_cols = []
for col in X_train.keys():
    feature_cols.append(tf.feature_column.numeric_column(key=col))
classifier = tf.estimator.DNNClassifier(feature_columns = feature_cols, hidden_units = [10, 10], n_classes = 2)


classifier.train(input_fn = lambda:train_input_fn(X_train, y_train, 100)
                , steps = 10000)

eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(X_test, y_test, 100))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))