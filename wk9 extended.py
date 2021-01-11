# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:19:13 2021

@author: belle
"""

#import the necessary libraries
import matplotlib.pyplot as plt
from tensorflow import keras

## import dataset from keras - MNIST - images: Test and Train Data
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()

plt.imshow(x_train[0,])

# pre-processing step: normalisation
x_train = x_train / 255.0
x_test = x_test / 255.0

# design the feedforward neural network
model = keras.models.Sequential()
model.add(keras.layers.Flatten()) # input layer for feeding input data 28*28=784 features/pixels into nn

model.add(keras.layers.Dense(100, activation='relu')) #hidden layer #1
model.add(keras.layers.Dense(100, activation='relu')) #hidden layer #2
model.add(keras.layers.Dense(100, activation='relu')) #hidden layer #3

model.add(keras.layers.Dense(10, activation = 'softmax')) #output layer

# train and validation of the designed neural network
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, validation_split=0.2, verbose=1)

# testing the model the best model out of trial and error)
loss_val, acc_val = model.evaluate(x_test, y_test)

print('Accuracy based on model validation is: ', acc_val*100)

y_predicted = model.predict_classes(x_test)

plt.imshow(x_test[0], cmap='grey')
plt.show()

print('The predicted class label by the model is: ', y_predicted[0])

Rec_Acc=0
for i in range(len(y_test)):
    if (y_predicted[i]==y_test[i]):
        Rec_Acc = Rec_Acc + 1
        
        
Rec_Acc= Rec_Acc/len(y_test)*100
print('Recognition Accuracy: ', Rec_Acc, '%')





