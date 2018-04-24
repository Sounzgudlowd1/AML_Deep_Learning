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

def get_data(num_patches = -1, rnd = True):
    image_patches = glob('../data/**/*.png', recursive = True)
    total_patches = len(image_patches)
    if num_patches ==  -1:
        num_patches = total_patches
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
    
        
    if rnd == True:
        return train_test_split(X, y, test_size = 0.2)
    else:
        return train_test_split(X, y, test_size = 0.2, random_state = 101085)
    

def plot_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.show()
    return
