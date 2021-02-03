# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:37:56 2021

@author: russe
"""

## importing the libraries required
import numpy as np
import pathlib
import pickle
import random
import matplotlib.pyplot as plt
import tensorflow as tf # using Tensorflow 2.4
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split #used to split data into training and test segments
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from tensorflow import keras #keras is the api that provides functionality to work with tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier



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

data        = [] # array with all list of images read from batch files
X           = [] # array with images data including channels (RGB)
y           = [] # array with labels (index of category of images)
# myModelFile = path + "anomaliesDetectionModel.h5" # file to save trained model
path =  "anomaliesDetectionModel.h5"

def CNN_model():
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




# Setting Hyperparameters
# noOfEpochs  = 100   # define number of epochs to execute
# myBatchSze  = 32  # size of each batch in interaction to get an epoch
# myTestSize  = 0.2 # how much have to be split for testing
# noOfFiles   = 5   # number of batch files to process
# myMinDelta  = 0.01# minimum improvement rate for do not early stop
# myPatience  = 2   # how many epochs run with improvement lower than myMinDelta 
# MyRandomSt  = 42  # random state for shuffling the data
# myMetric    = "accuracy" # type of metric used for training
# MyOptimizer = "adam"
# MyLoss      = "categorical_crossentropy"

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

path = pathlib.Path(__file__).resolve().parent # Getting the relative path of the file 
path = str(path) + "/"
# epochs    = 30
test_size = 0.2
number_of_batch_files = 5
myModelFile = path + "anomaliesDetectionModel.h5"
myMetric = "accuracy"

data      = [] # array with all images read from batch files
X         = [] # array with images data including channels (RGB)
y         = [] # array with labels (index of category of images)

# Loading data 
# images/ folder should be in the same location of the script

## data: data_batch_N dictionary
    # { 
    # b'labels': b'training batch 5 of 5' => title of dictionary
    # b'labels': [1, 8... n ] => array 1D of size 10,000 labels
    # b'data': array([[255, 252, 253,..] , [127, 126, 127, ...], [...]] ) => array of size 10,000 x 3,072 ( 1024 R + 1024 G + 1024 B )
    # b'filenames': [b'compact_car_s_001706.png', b'icebreaker_s_001689.png',...] => array of size 10,000 with files names
    # }    
for n in range(1,number_of_batch_files + 1,1):
    data = np.append(data, read( path + 'images/data_batch_' + str(n)) )

## meta dictionary:
    # {   
    # num_cases_per_batch': 10000
    #                      0           1            2       3        4       5        6        7         8         9
    # label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']
    # b'num_vis':    3072 ( 1024 R + 1024 G + 1024 B )   
    # } 
meta = read( path + 'images/batches.meta')

# Unifying all images pixels values in unique array X with 50,000 images
for images in data:
    if len(X) == 0:
        X = images[b'data'] # data[n]  => data = { data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch5   }
    else:
        X = np.concatenate((X, images[b'data']), axis=0)   # data [ [a,b,c,d ] ] 
    y = np.append(y, images[b'labels'] ) # y [ 0,1,2,3 ... 50,000] with 50,000 labels
  
[x_train, x_test, y_train, y_test] = train_test_split(X, y, test_size = test_size, random_state= 42 )
      
    
    # Pre-processing
x_trainNormalised = x_train / 255.0
x_testNormalised = x_test / 255.0

## Changing the shape of INPUT data
nInstancesTrain  = x_trainNormalised.shape[0]
nInstancesTest   = x_testNormalised.shape[0]
nRowns           = 32 
nColumns         = 32
nChannels        = 3  # 3 channels denotes one red, green and blue (RGB image)

#x_train = x_trainNormalised.reshape(8000, 32, 32, 3)
x_trainNormalised = x_trainNormalised.reshape(nInstancesTrain, nRowns, nColumns, nChannels) 

#x_test = x_test.x_testNormalised(2000, 32, 32, 3)
x_testNormalised = x_testNormalised.reshape(nInstancesTest, nRowns, nColumns, nChannels)

## Changing the shape of OUTPUT layer, also changing the labels of train and test into categorical data
# It creates hot vectors for the classes like: [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
y_trainCategorical = to_categorical(y_train)
y_testCategorical = to_categorical(y_test)

model= KerasClassifier(build_fn=CNN_model,verbose=0)

batch_size = [10,20,40,60]
epochs = [10,20,30,40]   
param_grid= dict(batch_size=batch_size , epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_trainNormalised, y_trainCategorical)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))