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

## Importing especific functions used in this program 
path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)
from functions import *

# initialising variables
data        = [] # array with all list of images read from batch files
X           = [] # array with images data including channels (RGB)
y           = [] # array with labels (index of category of images)
myModelFile = path + "anomaliesDetectionModel.h5" # file to save trained model

# Setting Hyperparameters
noOfEpochs  = 1   # define number of epochs to execute
myBatchSze  = 32  # size of each batch in interaction to get an epoch
myTestSize  = 0.2 # how much have to be split for testing
noOfFiles   = 5   # number of batch files to process
myMinDelta  = 0.05# minimum improvement rate for do not early stop
myPatience  = 2   # how many epochs run with improvement lower than myMinDelta 
MyRandomSt  = 42  # random state for shuffling the data
myMetric    = "accuracy" # type of metric used for training
MyOptimizer = "adam"
MyLoss      = "categorical_crossentropy"

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

# Selecting 40,000 instances for training set and 10,000 for test set
[x_train, x_test, y_train, y_test] = train_test_split(X, y, test_size = myTestSize, random_state= MyRandomSt )

print("\nShape of split data")
print("\nx_train initial shape:  ", x_train.shape)
print("x_test initial shape:  ", x_test.shape)
print("y_train initial shape:  ", len(y_train))
print("y_test initial shape:  ", len(y_test))
print("\n")
 
# Pre-processing to work with values between 0.0 and 1
x_trainNormalised = x_train / 255.0
x_testNormalised = x_test / 255.0

## Changing the shape of inputted data
nInstancesTrain  = x_trainNormalised.shape[0]
nInstancesTest   = x_testNormalised.shape[0]
nRowns           = 32 
nColumns         = 32
nChannels        = 3  # 3 channels denotes one red, green and blue (RGB image)

#x_trainNormalised= x_trainNormalised.reshape(   40,000,          32,     32,       3     )
x_trainNormalised = x_trainNormalised.reshape(nInstancesTrain, nRowns, nColumns, nChannels) 

#x_testNormalised = x_testNormalised.reshape(  10,000,        32,      32,       3     )
x_testNormalised = x_testNormalised.reshape(nInstancesTest, nRowns, nColumns, nChannels)

# Changing the shape of OUTPUT layer, also changing the labels of train and test into categorical data
# It creates hot vectors for the classes like: [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
y_trainCategorical = to_categorical(y_train)
y_testCategorical = to_categorical(y_test)

# Checking that train and test data categorical and normalised are the correct shape for NN
print("\nShape of normalised data")
print("\nx_train normalised shape: ", x_trainNormalised.shape)
print("x_test normalised shape: ", x_testNormalised.shape)
print("y_trainCategorical shape: ", y_trainCategorical.shape)
print("y_testCategorical shape: ", y_testCategorical.shape)
print("\n\n")
# Loading previously trained model  
## PENDIND: test if different model can be load and train with another number of layers, etc

# if files exist, load:
#    model = tf.keras.models.load_model(myModelFile)

"""  DROPOUT HINTS:
In practice, you can usually apply dropout only to the neurons in the top one to three layers (excluding the output layer).
(Aurelien pag. 481)

In the simplest case, each unit is retained with a fixed probability p 
independent of other units, where p can be chosen using a validation 
set or can simply be set at 0.5, which seems to be close to optimal for 
a wide range of networks and tasks. For the input units, however, 
the optimal probability of retention is usually closer to 1 than to 0.5.
(Dropout: A Simple Way to Prevent Neural Networks from Overfitting, 2014.)         
"""

# designing the Convolutional Neural Network 
model = tf.keras.models.Sequential()                                                                       #32    #32        #3           
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape = (nRowns, nColumns, nChannels), activation='relu')) #layer 1    
#model.add(tf.keras.layers.Dropout(0.0))        
# Size of Pooling of 2x2 is default for images
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2))) #, strides=2)) testing if strides improve accuracy
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')) # layer 2
model.add(tf.keras.layers.MaxPool2D(pool_size = (3,3)))

# designing by Fully Connect Neural Network
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))    
model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
## compile DNN => not sparse_categorical_crossentropy because classes are exclusives!
model.compile(optimizer=MyOptimizer, loss=MyLoss, metrics=[myMetric, f1_m, precision_m, recall_m, fbetaprecisionskewed, fbetarecallskewed])

# Stopping early according to myMinDelta to avoid overfitting. Trained model saved at myModelFile
myCallbacks = [EarlyStopping(monitor=myMetric, min_delta=myMinDelta , patience=myPatience, mode='auto'),
             ModelCheckpoint(filepath=myModelFile, monitor=myMetric, save_best_only=True, verbose=1)]
 
## Training the model according to the labels and chose hyperparameters
model.fit(x_trainNormalised, y_trainCategorical, epochs=noOfEpochs, batch_size=myBatchSze,  verbose=1, callbacks = myCallbacks)

## evaluating the accuracy using test data
loss_val, acc_val, f1_score, precision, recall, fbetaprecisionskewed, fbetarecallskewed = model.evaluate(x_testNormalised, y_testCategorical)
print('Accuracy is: ', acc_val)
print('F1 Score is: ', f1_score)
print('Precision is: ', precision)
print('Recall is: ', recall)
print('F- Beta (0.2) Score is: ', fbetaprecisionskewed)
print('F- Beta (2) Score is: ', fbetarecallskewed)

# Predicting aleatory sample from 0 to 10,000 (test set has 10,000 instances)
someSample = random.randint(0, (len(X)*myTestSize) - 1 ) 
y_predicted = np.argmax(model.predict(x_testNormalised), axis=-1)

print("y_predicted for test dataset index ",someSample," is ", meta[b'label_names'][y_predicted[someSample]])

# Print the image
printImage(x_test[someSample])
    
    
    
    
    
    
    
    
    
    

