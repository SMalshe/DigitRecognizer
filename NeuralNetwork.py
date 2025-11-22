import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

#Data loading
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Data normalization
x_train = x_train/255.0
x_test = x_test/255.0

#Creating the actual model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #Alternate way to use relu
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

#Training the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3)

#Saving and loading model
model.save('handwritten.keras')
model = tf.keras.models.load_model('handwritten.keras')

#Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test)
