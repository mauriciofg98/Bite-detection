from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
import os
import random
import time

from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.optimizers import Adam

NAME = "bite-detection-log-{}".format(int(time.time()))
tensorboard =TensorBoard(log_dir='logs/{}'.format(NAME))

#DATADIR = "C:/Users/Kowalski/PycharmProjects/ML/Bite-detection/"
DATADIR = "./"

CATEGORIES = ["normal", "overbite", "underbite"]
data = []
for category_num in range(3):
    category = CATEGORIES[category_num]
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (256, 256))
        data.append([new_array, category_num])

random.shuffle(data)

X = []
y = []
for features,label in data:
    X.append(features)
    y.append(label)

np.save('features.npy', X, allow_pickle=True)
X = np.load('features.npy', allow_pickle=True)

X = X.reshape((-1, 256, 256, 1))


trainX = np.delete(X, np.s_[0:10], axis=0)
trainY = np.delete(y, np.s_[0:10], axis=0)
testX = X[0:10]
testY = y[0:10]


model2 = Sequential()
model2.add(Conv2D(200, (3, 3), activation='relu', input_shape=X.shape[1:]))
model2.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model2.add(MaxPooling2D((5, 5)))
model2.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model2.add(Conv2D(140,kernel_size=(3,3),activation='relu'))
model2.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
model2.add(Conv2D(50,kernel_size=(3,3),activation='relu'))
model2.add(MaxPool2D(5,5))

model2.add(Flatten())
model2.add(Dense(180, activation='relu'))
model2.add(Dense(100,activation='relu'))
model2.add(Dense(50,activation='relu'))
model2.add(Dropout(rate=0.5))
model2.add(Dense(6, activation='softmax'))

model2.compile(optimizer=Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
'''model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])'''
model2.fit(trainX, trainY, epochs=3, validation_split=0.3, callbacks =[tensorboard])

test_loss, test_acc = model2.evaluate(testX, testY, verbose=2)
print('\nTest accuracy:', test_acc)