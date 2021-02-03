#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General functions 
Date: 03/02/2021
Author: Diego Bueno (ID: 23567850) / Isabelle Sypott (ID: 21963427 )
e-mail: d.bueno.da.silva.10@student.scu.edu.au / i.sypott.10@student.scu.edu.au

General functions used for training and deploying anamalies detection program

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

""" Function read( file_name  )

    Read a pickle file format  
    and return a Python dictionary with its content.

    parameters: (String) file_name

    return: 
        dict: a dictionary with the content encoding in bytes
    
"""
def read(file_name):
    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



""" Function to get formulas for precision, recall and f1 values"""

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision



def f1_m(y_true, y_pred): ## has a beta of 1 as it equally weights recall and precision
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def fbetaprecisionskewed(y_true, y_pred, threshold_shift=0.5):
    beta = 0.2 #Beta value below 1 favours precision. 

    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 
    #y_pred = K.round(y_pred + threshold_shift)

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall) 



def fbetarecallskewed(y_true, y_pred, threshold_shift=0.5):
    beta = 2 #Beta value greater than 1 favours recall. 

    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 
    #y_pred = K.round(y_pred + threshold_shift)

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall) 

    ##Not using log loss as it is only used for binary classification problems 
    #def log loss 
    #log_loss= log_loss(y_test,y_train)
    #return log_loss




def printImage(imageArray):
    
    im_r = imageArray[0:1024].reshape(32, 32)
    im_g = imageArray[1024:2048].reshape(32, 32)
    im_b = imageArray[2048:3072].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    plt.imshow(img) 
    plt.show()



#can't get this functional either I'm getting mad
# plt.plot(epochs, acc_val)
# plt.show()
# plt.plot(epochs, f1_score)
# plt.plot(epochs, precision)
# plt.plot(epochs, recall)
