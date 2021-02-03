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
import tensorflow as tf # using Tensorflow 2.4
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 


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



""" Function getMyEarlyStop( myMonitor, myPatience,  myModelFile )

    Set callback functions to early stop training, save the best model  
    to disk and return it as array.

    parameters: (String) myMonitor - metric name chose for monitor
                   (int) myPatience - number of epochs to interrupt in case 
                                      there is no longer improvement
                (String) myModelFile - path including file name to save the 
                                      best model.

    return: 
        callbacks: array with callback functions
    
"""
def getMyEarlyStop( myMonitor = "", myPatience = 0, myModelFile = "" ):
    
    if not myMonitor or myPatience <= 0 or not myModelFile:
        print("Invalid parameters!")
        return []
    
    callbacks = [EarlyStopping(monitor=myMonitor, patience=myPatience, mode='auto'),
    ModelCheckpoint(filepath=myModelFile, monitor=myMonitor, save_best_only=True, verbose=1)]
    return callbacks                 



""" Function getMyModel(layers<beta>, dropout)

    Create a customised model according with parameters
    and return a model object.

    parameters: (Array) layers  - <beta>
    
                (Array) dropout - array with dropout rate after each layer
                                  null for it does not apply droupout and float for applying.
                                 Example: [null, null, null, 0.5, null, 0.25] will apply
                                          0.5 dropout rate at 5th layer and 
                                          0.25 dropout rate at 7th layer.
                                          Invalid number layer is ignored.
                                          OBS: it does not include input and output layers

    return: 
        model: a compiled model according to parameters.
    
"""

def getMyModel(nLayers, nRowns, nColumns, nChannels, MyDropout = [] ):

    # designing the Convolutional Neural Network 
    model = tf.keras.models.Sequential()
    
    # CNN
                                                                                   #32    #32        #3           
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape = (nRowns, nColumns, nChannels), activation='relu'))     
    if len(MyDropout) > 0 and MyDropout[0]:
        model.add(tf.keras.layers.Dropout(MyDropout[0]))    
    
    # Size of Pooling of 2x2 is default for images
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if len(MyDropout) > 1 and MyDropout[1]:
        model.add(tf.keras.layers.Dropout(MyDropout[1]))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')) # layer 2
    if len(MyDropout) > 2 and MyDropout[2]:
        model.add(tf.keras.layers.Dropout(MyDropout[2]))

    model.add(tf.keras.layers.MaxPool2D(pool_size = (3,3)))
    if len(MyDropout) > 3 and MyDropout[3]:
        model.add(tf.keras.layers.Dropout(MyDropout[3]))
    
    # Classifing by Fully Connect Neural Network
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    if len(MyDropout) > 4 and MyDropout[4]:
        model.add(tf.keras.layers.Dropout(MyDropout[4]))
    
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    ## compile CNN => not sparse_categorical_crossentropy because classes are exclusives!
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1_m, precision_m, recall_m])

    return model


""" Function to get customised formulas for precision, recall and f1 values"""
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



""" Function printImage( imageArray  )

    Print a image according to its pixels values.
    
    parameters: (array) imageArray

    return: 
        none
    
"""
def printImage(imageArray):
    
    im_r = imageArray[0:1024].reshape(32, 32)
    im_g = imageArray[1024:2048].reshape(32, 32)
    im_b = imageArray[2048:3072].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    plt.imshow(img) 
    plt.show()



""" Function plotChatEpochsVsMetris(   )

    Plet a chat showing Epochs versus Metris
    
    parameters:

    return: 
        none
    
"""
def plotChatEpochsVsMetris(epochs,acc_val, f1_score,precision,recall):
    
    plt.plot(epochs, acc_val)
    plt.plot(epochs, f1_score)
    plt.plot(epochs, precision)
    plt.plot(epochs, recall)
    plt.show()
    


""" Function saveResultToFile(   )

    Save results
    
    parameters:

    return: 
        none
    
"""
def saveToFile(file , contend ):

    try:
        f = open(file, "a")
        f.write(contend)
        f.close()
    except:
        print("\nError to save contend " + contend + " to file " + file + "!" )

    


