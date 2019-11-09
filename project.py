from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import random

DATADIR = "C:/Users/Kowalski/PycharmProjects/ML/Bite-detection/"
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


model = Sequential([
    Conv2D(64, (2,2), input_shape=X.shape[1:]),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (2,2), input_shape=X.shape[1:]),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(64),
    Dense(1),
    Activation('sigmoid', input_shape=X.shape[1:])

])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(trainX, trainY, epochs=10, batch_size=1, validation_split=0.1)

test_loss, test_acc = model.evaluate(testX, testY, verbose=2)
print('\nTest accuracy:', test_acc)