# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:14:16 2021

@author: russe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 01:00:12 2021

Script for simulating all possible Hyperparameters 
Date: 04/02/2021
Author: Diego Bueno (ID: 23567850) / Isabelle Sypott (ID: 21963427 )
e-mail: d.bueno.da.silva.10@student.scu.edu.au / i.sypott.10@student.scu.edu.au


"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
## Importing especific functions used in this program 
path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)
from findingTheBestHyperparameters import *
from functions import *


# Change the name of the file if you want to see separately

resultsFile   = "resultsByVaringbatchSize.csv"

## You can define any array with possible values and run

testSizeToSimulate = [0.15,0.18,0.2,0.22,0.25]


# Change the loop and the attribution, in this case on line 42 to check results

for attempt in range(0,len(testSizeToSimulate)):
    
    print("\nSimulating with test size " + str(testSizeToSimulate[attempt]) + "...\n")
  
    # Setting Hyperparameters
    noOfEpochs  = 11   # define number of epochs to execute
    myBatchSze  = 96  # size of each batch in interaction to get an epoch
    myTestSize  = testSizeToSimulate[attempt] # how much have to be split for testing
    noOfFiles   = 5   # number of batch files to process
    myMinDelta  = 0.05# minimum improvement rate for do not early stop
    myPatience  = 2   # how many epochs run with improvement lower than myMinDelta 
    MyRandomSt  = 42  # random state for shuffling the data
    myMetric    = "accuracy" # type of metrics used 
    MyOptimizer = "adam"
    MyLoss      = "categorical_crossentropy"
    MyLearnRate = 0 # 0 value will keep default. Eg. adam => 0.001 
    noLayersCNN = 3
    noFiltersCNN= 32 # it will increase by plus 32 for each hidden layer
    hiddenActCNN= 'relu'
    outputActCNN= 'relu'
    dropOutsCNN = []
    noLayersFNN = 2
    noNeuronsFNN= 10 # output layer. It will multly for each hidden layer 
    hiddenActFNN= 'relu'
    outputActFNN= 'softmax'
    dropOutsFNN = []
    
    
    thisModel = findingTheBestHyperparameters(resultsFile,                            
        noOfEpochs, 
        myBatchSze, 
        myTestSize, 
        noOfFiles,  
        myMinDelta, 
        myPatience, 
        MyRandomSt, 
        myMetric,  
        MyOptimizer,
        MyLoss,     
        MyLearnRate,
        noLayersCNN, 
        noFiltersCNN,
        hiddenActCNN,
        outputActCNN,
        dropOutsCNN,
        noLayersFNN, 
        noNeuronsFNN, 
        hiddenActFNN,
        outputActFNN,
        dropOutsFNN 
        )

     
# reading the results and plot the comparison with accuracy
resultsDf = pd.read_csv(path + resultsFile)

resultsDf.columns = [
"TRAINING_SIZE", "TEST_SIZE", "RANDOM_STATE", "N_EPOCHS", "N_LAYERS_CNN",
"TOTAL_FILTERS_CNN", "HIDDEN_ACTIVATIONS_CNN", "OUTPUT_ACTIVATION_CNN",
"LIST_OF_DROPOUTS_CNN", "N_LAYERS_FNN" , "TOTAL_NEURONS_FNN", 
"HIDDEN_ACTIVATIONS_CNN" , "OUTPUT_ACTIVATIONS_CNN", "DROPOUTS_FNN",
"BATCHSIZE", "MINDELTA" , "PATIENCE" , "METRIC" ,  "OPTMIZER",
"LEARN_RATE", "TYPE_OF_LOSS" ,  "LOSS_VALUE" , "ACCURACY", "F1_SCORE" , 
"PRECISION", "RECALL" , "F-BETA02", "F-BETA2"]

resultsDf = resultsDf.astype({"TEST_SIZE": float})
    
# Define llenDf colours 
jet = plt.get_cmap('jet')
colors = iter(jet(np.linspace(0,1,len(resultsDf.index))))

plt.title('Comparision Accuracy vs Test size')    
plt.xlabel("Test Size") 
plt.ylabel("Accuracy" )

for index, row in resultsDf.iterrows():
   # Plotting values in different colours
   plt.scatter(row["TEST_SIZE"],row["ACCURACY"], color= next(colors) )

plt.show()
        
   