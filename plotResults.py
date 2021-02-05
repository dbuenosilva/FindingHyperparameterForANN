#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:19:15 2021

@author: diego
"""

import sys
import pathlib
## Importing especific functions used in this program 
path = str(pathlib.Path(__file__).resolve().parent) + "/"
sys.path.append(path)
from findingTheBestHyperparameters import *
from functions import *

# Inform the file name of CSV you would like to see

#csvFile = "resultsByVarinNoofN_LAYERS_FNNSize.csv"
#csvFile = "resultsByVaringOPTMIZERSize.csv"
#csvFile = "resultsByVaringbatchSize.csv"
#csvFile = "resultsByVaringTestSize.csv"
#csvFile = "resultsByVaringPATIENCESize.csv"
#csvFile = "resultsByVarinNoofLayersCNNSize.csv"
#csvFile = "resultsByVarinHIDDEN_ACTIVATIONS_FNN.csv"
#csvFile = "resultsByVarinNoofHIDDEN_ACTIVATIONS_CNN.csv"
#csvFile = "resultsByVarinNoofOUTPUT_ACTIVATION_CNN.csv"
#csvFile = "resultsByVarinOUTPUT_ACTIVATIONS_FNN.csv"
#csvFile = "resultsByVaryingEpochs.csv"
csvFile = "bestSolution.csv"


"""
    Change the labels and colunm names
    
    Possible colunm names:

    "TRAINING_SIZE", "TEST_SIZE", "RANDOM_STATE", "N_EPOCHS", "N_LAYERS_CNN",
    "TOTAL_FILTERS_CNN", "HIDDEN_ACTIVATIONS_CNN", "OUTPUT_ACTIVATION_CNN",
    "LIST_OF_DROPOUTS_CNN", "N_LAYERS_FNN" , "TOTAL_NEURONS_FNN", 
    "HIDDEN_ACTIVATIONS_FNN" , "OUTPUT_ACTIVATIONS_FNN", "DROPOUTS_FNN",
    "BATCHSIZE", "MINDELTA" , "PATIENCE" , "METRIC" ,  "OPTMIZER",
    "LEARN_RATE", "TYPE_OF_LOSS" ,  "LOSS_VALUE" , "ACCURACY", "F1_SCORE" , 
    "PRECISION", "RECALL" , "F-BETA02", "F-BETA2"
    
"""

""" Scartter chart between two numerical colunms """
plotScatterChatResultsComparisson(path + csvFile,"TEST_SIZE","ACCURACY",
                           "KFold", "Accuracy",1) # 1 plot mean; 0 dont plot

    
""" Bar chart chart between one numerical colunm and categorical column """

#plotBarChatResultsComparisson(path + csvFile,"OUTPUT_ACTIVATION_CNN","ACCURACY",
#                           "Output Activation Function CNN", "Accuracy")

