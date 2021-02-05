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
import pandas as pd
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



""" Function getMyModel( *args )

    Create a customised model according with parameters
    and return a model object.

    parameters:(list) inputShape, 
               (string) myMetric,
               (string) MyOptimizer,
               (string) MyLoss,
               (float) MyLearnRate,
               (int) noOfLayersCNN, 
               (int) noFiltersCNN,
               (string) hiddenActCNN,
               (string) outputActCNN,
               (array) dropOutsCNN,
               (int) noLayersFNN,
               (int) noNeuronsFNN,
               (string) hiddenActFNN,
               (string) outputActFNN,
               (array) dropOutsFNN - array with dropout rate after each layer
                                  null for it does not apply droupout and float for applying.
                                 Example: [null, null, null, 0.5, null, 0.25] will apply
                                          0.5 dropout rate at 5th layer and 
                                          0.25 dropout rate at 7th layer.
                                          Invalid number layer is ignored.
                                          OBS: it does not include input and output layers

    return: 
        model: a compiled model according to parameters.
    
"""

def getMyModel(inputShape, 
               myMetric,
               MyOptimizer,
               MyLoss,
               MyLearnRate,
               noOfLayersCNN, 
               noFiltersCNN,
               hiddenActCNN,
               outputActCNN,
               dropOutsCNN,
               noLayersFNN,
               noNeuronsFNN,
               hiddenActFNN,
               outputActFNN,
               dropOutsFNN):


    model = tf.keras.models.Sequential()

    """ Designing the Convolutional Neural Network in case noOfLayersCNN > 0  """ 

    for i in range(1,noOfLayersCNN):  
        
        if i == 1: # informing input shape
            model.add(tf.keras.layers.Conv2D(filters=noFiltersCNN*i, kernel_size=(3,3), input_shape = inputShape, activation=hiddenActCNN, padding='same'))     
            model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2))) # Size of Pooling of 2x2 is default for images
            # DO NOT DROP OUT INPUTS
        else:
            model.add(tf.keras.layers.Conv2D(filters=noFiltersCNN*i, kernel_size=(3,3), activation=hiddenActCNN, padding='same'))                 
            model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2))) # Size of Pooling of 2x2 is default for images
            # Applying dropouts for the layer        
            if len(dropOutsCNN) > i and dropOutsCNN[i]:
                model.add(tf.keras.layers.Dropout(dropOutsCNN[i]))    

    # CNN output layer
    model.add(tf.keras.layers.Conv2D(filters=noFiltersCNN * i, kernel_size=(3,3), activation=outputActCNN, padding='same'))                 
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2))) # Size of Pooling of 2x2 is default for images
    # DO NOT DROP OUT CNN OUPUTS


    """ Designing the Fully Connect Neural Network in case noLayersFNN > 0 """
    
    if noLayersFNN > 0:
        model.add(tf.keras.layers.Flatten())
        # DO NOT DROP OUT INPUTS

    for i in range(1,noLayersFNN):  
        model.add(tf.keras.layers.Dense(noNeuronsFNN * 10 * (noLayersFNN + 1 - i), activation=hiddenActFNN))
        print("No Neurons in layer " + str(i) + ": ", noNeuronsFNN * 10 * (noLayersFNN + 1 - i) )
        if len(dropOutsFNN) > i and dropOutsFNN[i]: # Applying dropouts for the layer        
            model.add(tf.keras.layers.Dropout(dropOutsFNN[i]))        
    
    model.add(tf.keras.layers.Dense(noNeuronsFNN, activation=outputActFNN))
    # DO NOT DROP OUT FNN OUPUTS

    """ Compiling the Deep Neural Network """
    
    model.compile(optimizer=MyOptimizer, loss=MyLoss, metrics=[myMetric, f1_m, precision_m, recall_m, fbetaprecisionskewed, fbetarecallskewed])

    # Setting custom learn rate
    if MyLearnRate > 0:
        model.optimizer.lr = MyLearnRate
        
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



""" Function plotScatterChatResultsComparisson(   )

    Plot a chart showing Hyperparameters versus its performance
    
    parameters: results - results CSV file
                x - column name of x axis
                y - column name of values of y axis
                title - title of the chart
                xLabel - description of label x
                yLabel - description of label y

    return: 
        none
    
"""
def plotScatterChatResultsComparisson(results,x,y,xLabel, yLabel,mean):
    
    # reading the results and plot the comparison with accuracy
    resultsDf = pd.read_csv(results)
    
    resultsDf.columns = [
    "TRAINING_SIZE", "TEST_SIZE", "RANDOM_STATE", "N_EPOCHS", "N_LAYERS_CNN",
    "TOTAL_FILTERS_CNN", "HIDDEN_ACTIVATIONS_CNN", "OUTPUT_ACTIVATION_CNN",
    "LIST_OF_DROPOUTS_CNN", "N_LAYERS_FNN" , "TOTAL_NEURONS_FNN", 
    "HIDDEN_ACTIVATIONS_FNN" , "OUTPUT_ACTIVATIONS_FNN", "DROPOUTS_FNN",
    "BATCHSIZE", "MINDELTA" , "PATIENCE" , "METRIC" ,  "OPTMIZER",
    "LEARN_RATE", "TYPE_OF_LOSS" ,  "LOSS_VALUE" , "ACCURACY", "F1_SCORE" , 
    "PRECISION", "RECALL" , "F-BETA02", "F-BETA2"]
        
    # Define llenDf colours 
    jet = plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0,1,len(resultsDf.index))))
    
    plt.title('Comparision ' + yLabel + ' vs ' + xLabel)    
    plt.xlabel(xLabel) 
    plt.ylabel(yLabel)
    
    for index, row in resultsDf.iterrows():
       # Plotting values in different colours
       plt.scatter(row[x],row[y], color= next(colors) )
     
    avg = np.mean(resultsDf[y])    
     
    if mean == 1:
          # plot the ideal case in red color
          plt.plot(resultsDf[x], [avg for _ in range(len(resultsDf[x]))], color='red')
    
    
    plt.show()
            

""" Function plotBarChatResultsComparisson(   )

    Plot a bar chart showing categorical Hyperparameters versus its performance
    
    parameters: results - results CSV file
                x_names - array with column of category names of x axis
                y - column name of values of y axis
                title - title of the chart
                xLabel - description of label x
                yLabel - description of label y

    return: 
        none
    
"""
def plotBarChatResultsComparisson(results,x,y,xLabel, yLabel):

    # reading the results and plot the comparison with accuracy
    resultsDf = pd.read_csv(results)
    
    resultsDf.columns = [
    "TRAINING_SIZE", "TEST_SIZE", "RANDOM_STATE", "N_EPOCHS", "N_LAYERS_CNN",
    "TOTAL_FILTERS_CNN", "HIDDEN_ACTIVATIONS_CNN", "OUTPUT_ACTIVATION_CNN",
    "LIST_OF_DROPOUTS_CNN", "N_LAYERS_FNN" , "TOTAL_NEURONS_FNN", 
    "HIDDEN_ACTIVATIONS_FNN" , "OUTPUT_ACTIVATIONS_FNN", "DROPOUTS_FNN",
    "BATCHSIZE", "MINDELTA" , "PATIENCE" , "METRIC" ,  "OPTMIZER",
    "LEARN_RATE", "TYPE_OF_LOSS" ,  "LOSS_VALUE" , "ACCURACY", "F1_SCORE" , 
    "PRECISION", "RECALL" , "F-BETA02", "F-BETA2"]

    x_pos = [i for i, _ in enumerate(resultsDf[x])]
    plt.xticks(x_pos, resultsDf[x])
    plt.bar(x_pos, resultsDf[y], color='green')         
    plt.title('Comparision ' + yLabel + ' vs ' + xLabel)    
    plt.xlabel(xLabel) 
    plt.ylabel(yLabel)        
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



