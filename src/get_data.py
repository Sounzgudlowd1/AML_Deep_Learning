# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:08:16 2018

@author: Erik
"""
from glob import glob
import cv2
import fnmatch
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import random as rand


def get_file_paths(num_files = -1, randomly_sample = True):
    image_patches = glob('../data/**/*.png', recursive = True)
    if(num_files == -1):
        return image_patches
        
    out_patches = []
    if randomly_sample:
        i = 0
        while(i < num_files):
            out_patches.append(image_patches[rand.randint(0, len(image_patches) -1)])
            i = i + 1
    else:
        out_patches = image_patches[:num_files]
    
    return out_patches
            

def get_data(image_patches, randomly_split = True):
    num_patches = len(image_patches)
        
    pattern_zero = '*class0.png'
    pattern_one = '*class1.png'
    classZero = fnmatch.filter(image_patches, pattern_zero)
    classOne = fnmatch.filter(image_patches, pattern_one)
    X = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    for img in image_patches[:num_patches]:
        full_size_image = cv2.imread(img)
        X.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    if randomly_split == True:
        return train_test_split(X, y, test_size = 0.2)
    else:
        return train_test_split(X, y, test_size = 0.2, random_state = 101085)

def convert_to_dataframe(X, y):
    cols = []
    for i in range(7500):
        cols.append('f' + str(i))

    X = np.array(X)/255.0
    X = X.reshape(len(X), 50 * 50 * 3) 
    X_df = pd.DataFrame(columns = cols, data = X)
    
    y = np.array(y)
    y_df = pd.DataFrame(columns = ['IDC (+)'], data = y)
    
    return X_df, y_df


def plot_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.show()
    return


