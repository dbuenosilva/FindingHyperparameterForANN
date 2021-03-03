#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomalies Detection on images
Date: 24/01/2021
Author: Diego Bueno (ID: 23567850) / Isabelle Sypott (ID: 21963427 )
e-mail: d.bueno.da.silva.10@student.scu.edu.au / i.sypott.10@student.scu.edu.au

"""

## importing the libraries required
import numpy as np
import sys
import pathlib
import random
import tensorflow as tf # using Tensorflow 2.4
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split #used to split data into training and test segments
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from sklearn.model_selection import KFold

## Importing especific functions used in this program 
path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)
from functions import *

# initialising variables
data        = [] # array with all list of images read from batch files
X           = [] # array with images data including channels (RGB)
y           = [] # array with labels (index of category of images)
myModelFile = path + "anomaliesDetectionModel.h5" # file to save trained model
myResults   = path + "3kfoldera.csv" 

# Setting Hyperparameters
noOfEpochs  = 20  # define number of epochs to execute
myBatchSze  = 128 # size of each batch in interaction to get an epoch
myTestSize  = 0.20# how much have to be split for testing
noOfFiles   = 5   # number of batch files to process
myMinDelta  = 0.01# minimum improvement rate for do not early stop
myPatience  = 2   # how many epochs run with improvement lower than myMinDelta 
MyRandomSt  = 42  # random state for shuffling the data
myMetric    = "accuracy" # type of metric used for training
MyOptimizer = "adam"
MyLoss      = "categorical_crossentropy"
MyNumKfolds = 3
totalAcc    = []
totalLoss   = []
    

# Loading the data. "images/" folder must be in the same location of the script
#
# Structure of meta dictionary
#
## meta dictionary:
    # {   
    # num_cases_per_batch': 10000
    #                      0           1            2       3        4       5        6        7         8         9
    # label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
    # b'num_vis':    3072 ( 1024 R + 1024 G + 1024 B )   
    # } 
meta = read( path + 'images/batches.meta')

#
# Structure of data dictionary
#
## data: data_batch_N dictionary
    # { 
    # b'labels': b'training batch 5 of 5' => title of dictionary
    # b'labels': [1, 8... n ] => array 1D of size 10,000 labels
    # b'data': array([[255, 252, 253,..] , [127, 126, 127, ...], [...]] ) => array of size 10,000 x 3,072 ( 1024 R + 1024 G + 1024 B )
    # b'filenames': [b'compact_car_s_001706.png', b'icebreaker_s_001689.png',...] => array of size 10,000 with files names
    # }        
for n in range(1,noOfFiles + 1,1):
    data = np.append(data, read( path + 'images/data_batch_' + str(n)) )

# Unifying all images pixels values in unique array X with 50,000 images
for images in data: # data = { data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch5   }
    if len(X) == 0:
        X = images[b'data'] 
    else: # Need to concatenate instead append to create a unique array with 50,000 images
        X = np.concatenate((X, images[b'data']), axis=0)
    y = np.append(y, images[b'labels'] ) # y [ 0,1,2,3 ... 50,000] with 50,000 labels

print("Shape of inputted data")        
print("\nX shape with all images data: ", X.shape)
print("y shape with all images labels", y.shape)

# Pre-processing to work with values between 0.0 and 1
x_Normalised = X / 255.0

## Changing the shape of inputted data
nInstancesTrain  = x_Normalised.shape[0]
nRowns           = 32 
nColumns         = 32
nChannels        = 3  # 3 channels denotes one red, green and blue (RGB image)

#x_trainNormalised= x_trainNormalised.reshape( 50,000,  32,     32,       3     )
x_Normalised = x_Normalised.reshape(nInstancesTrain, nRowns, nColumns, nChannels) 

# Changing the shape of OUTPUT layer, also changing the labels of train and test into categorical data
# It creates hot vectors for the classes like: [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
y_categorical = to_categorical(y)
 
## Loading previously trained model
# If the model has been trained before, load it
try:
    model = tf.keras.models.load_model(myModelFile)
#            custom_objects={'f1_m': f1_m,
#                            'precision_m': precision_m, 
#                            'recall_m': recall_m, 
#                            'fbetaprecisionskewed': fbetaprecisionskewed, 
#                            'fbetarecallskewed': fbetarecallskewed})
    print("Model file " + myModelFile + " sucessfully loaded!\n")
    
except OSError:
    print("Previously saved model does not exist.\n" +
          "Creating first file " + myModelFile + "\n")
except:
    print("Error trying to open the file " + myModelFile +
          "\nCurrent file will be replaced!\n")
 
    
    
# Define the K-fold Cross Validator insted of split test set
kfold = KFold(n_splits=MyNumKfolds, shuffle=True)

foldNumber = 1
for train, test in kfold.split(x_Normalised, y_categorical):

    print('\nTraining folder ',foldNumber,'\n')

    ## Deploying the best configuration for Deep Neural Networking
    #  found by findingTheBestHyperparameters.py
    
    # designing the Convolutional Neural Network 
    model = tf.keras.models.Sequential() 
                                                  #32    #32        #3           
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape = (nRowns, nColumns, nChannels), activation='relu'))           
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2))) #, strides=2)) testing if strides improve accuracy
    #model.add(tf.keras.layers.Dropout(0.8)) 
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')) 
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    #model.add(tf.keras.layers.Dropout(0.5)) 
    
#    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(3,3), activation='relu', padding='same'))
#    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    #model.add(tf.keras.layers.Dropout(0.4))
    
#    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
#    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    #model.add(tf.keras.layers.Dropout(0.5))
    
    
    # designing by Fully Connect Neural Network
    model.add(tf.keras.layers.Flatten())
#    model.add(tf.keras.layers.Dense(256, activation='relu'))    
#    model.add(tf.keras.layers.Dense(128, activation='relu'))    
    model.add(tf.keras.layers.Dense(32, activation='relu'))    
    model.add(tf.keras.layers.Dense(16, activation='relu'))   
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
    ## compile DNN => not sparse_categorical_crossentropy because classes are exclusives!
    model.compile(optimizer=MyOptimizer, loss=MyLoss, metrics=[myMetric])
    
    # Stopping early according to myMinDelta to avoid overfitting. Trained model saved at myModelFile
    myCallbacks = [EarlyStopping(monitor=myMetric, min_delta=myMinDelta , patience=myPatience, mode='auto'),
                 ModelCheckpoint(filepath=myModelFile, monitor=myMetric, save_best_only=True, verbose=1)]
     
    ## Training the model according to the labels and chose hyperparameters
    model.fit(x_Normalised[train], y_categorical[train], epochs=noOfEpochs, batch_size=myBatchSze,  verbose=1, callbacks = myCallbacks)
    
    
    ## evaluating the accuracy using test data
#    loss_val, acc_val, f1_score, precision, recall, fbetaprecisionskewed, fbetarecallskewed = model.evaluate(x_Normalised[test], y_categorical[test])
    loss_val, acc_val  = model.evaluate(x_Normalised[test], y_categorical[test])
    print('\n\nResults for folder ',foldNumber)
    print('\nAccuracy for folder: ', acc_val)
    print('Loss: ', loss_val)
#    print('F1 Score: ', f1_score)
#    print('Precision: ', precision)
#    print('Recall: ', recall)
#    print('F- Beta (0.2) Score: ', fbetaprecisionskewed)
#    print('F- Beta (2) Score: ', fbetarecallskewed)
    
    foldNumber += 1
    
    totalAcc.append(acc_val)
    totalLoss.append(loss_val)
    
    
    """ Saving results to csv file for analyse"""
    
    #
    # Results file layout:
    #
    #       1           2             3          4            5 
    # TRAING SIZE ; TEST SIZE ; RANDOM STATE; N. EPOCHS ; N. LAYERS CNN ;
    #
    #        6                    7                        8
    # TOTAL FILTERS CNN ; HIDDEN ACTIVATIONS CNN ; OUTPUT ACTIVATION CNN;
    #
    #       9                     10                11
    # LIST OF DROPOUTS CNN ; N. LAYERS FNN ; TOTAL NEURONS FNN; 
    #     
    #      12                        13                    14
    # HIDDEN ACTIVATIONS CNN ; OUTPUT ACTIVATIONS CNN; LIST OF DROPOUTS FNN
    #
    #      15             16            17           18          19
    # BATCH SIZE ; MIN DELTA ES ; PATIENCE ES ; METRIC ;       OPTMIZER ;
    #
    #     20              21             22          23        24
    # LEARNING RATE ; TYPE OF LOSS ;  LOSS VALUE ; ACCURACY ; F1 SCORE ; 
    #
    #   25        26           27          28
    # PRECISION ; RECALL ; F- BETA 0.2 ; F- BETA 2
    #
    
    contend = str(len(X))
    contend += "," + str(foldNumber)
    contend += "," + str(MyRandomSt)
    contend += "," + str(myTestSize)
    contend += "," + str(4)
    contend += "," + str(32)
    contend += "," + "relu"
    contend += "," + "relu"
    contend += "," + ""
    contend += "," + str(5)
    contend += "," + str(10)
    contend += "," + "relu"
    contend += "," + "softmax"
    contend += "," + ""
    contend += "," + str(myBatchSze)
    contend += "," + str(myMinDelta)
    contend += "," + str(myPatience)
    contend += "," + myMetric
    contend += "," + MyOptimizer
    contend += "," + str(0)
    contend += "," + MyLoss     
    contend += "," + str(loss_val)
    contend += "," + str(acc_val)
    contend += "," + str(0)
    contend += "," + str(0) 
    contend += "," + str(0) 
    contend += "," + str(0) 
    contend += "," + str(0) 
    contend += "\n"
    
    saveToFile(myResults, contend)
    

print("Mean accuraccy for all folders: ", np.mean(totalAcc))
print("Mean loss for all folders: ", np.mean(totalLoss))

# Predicting aleatory sample from 0 to 10,000 (test set has 10,000 instances)
#someSample = random.randint(0, (len(X)*myTestSize) - 1 ) 
#y_predicted = np.argmax(model.predict(x_testNormalised), axis=-1)

#print("y_predicted for test dataset index ",someSample," is ", meta[b'label_names'][y_predicted[someSample]])

# Print the image
#printImage(x_test[someSample])
    
