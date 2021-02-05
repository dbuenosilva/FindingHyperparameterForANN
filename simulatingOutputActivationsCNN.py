#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 01:00:12 2021

Script for simulating all possible Hyperparameters 
Date: 04/02/2021
Author: Diego Bueno (ID: 23567850) / Isabelle Sypott (ID: 21963427 )
e-mail: d.bueno.da.silva.10@student.scu.edu.au / i.sypott.10@student.scu.edu.au


"""
import sys
import pathlib
## Importing especific functions used in this program 
path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)
from findingTheBestHyperparameters import *
from functions import *


# Change the name of the file if you want to see separately

resultsFile   = "resultsByVaringCNNOutput.csv"

## You can define any array with possible values and run

valuesToSimulate = ["softmax","relu"]   

# Change the loop and the attribution, in this case on line 42 to check results

for attempt in range(0,len(valuesToSimulate)):
    
    print("\nSimulating with activation CNN " + str(valuesToSimulate[attempt]) + "...\n")
  
    # Setting Hyperparameters
    noOfEpochs  = 9   # define number of epochs to execute
    myBatchSze  = 128 # size of each batch in interaction to get an epoch
    myTestSize  = 0.25# how much have to be split for testing
    noOfFiles   = 5   # number of batch files to process
    myMinDelta  = 0.01 # minimum improvement rate for do not early stop
    myPatience  = 2   # how many epochs run with improvement lower than myMinDelta 
    MyRandomSt  = 42  # random state for shuffling the data
    myMetric    = "accuracy" # type of metrics used 
    MyOptimizer = "adam"
    MyLoss      = "categorical_crossentropy"
    MyLearnRate = 0 # 0 value will keep default. Eg. adam => 0.001 
    noLayersCNN = 4
    noFiltersCNN= 32 # it will increase by plus 32 for each hidden layer
    hiddenActCNN= 'relu'  #
    outputActCNN= valuesToSimulate[attempt]
    dropOutsCNN = []
    noLayersFNN = 5
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
        
   