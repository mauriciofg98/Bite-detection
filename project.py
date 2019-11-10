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
from tensorflow.python.keras.optimizers import Adam, rmsprop

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

'''plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i], cmap=plt.cm.binary)
    plt.xlabel(CATEGORIES[y[i]])
plt.show()'''
np.save('features.npy', X, allow_pickle=True)
X = np.load('features.npy', allow_pickle=True)

X = X.reshape((-1, 256, 256, 1))


trainX = np.delete(X, np.s_[0:10], axis=0)
trainY = np.delete(y, np.s_[0:10], axis=0)
testX = X[0:10]
testY = y[0:10]


'''
MODEL #1 TEST

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

END MODEL #1

MODEL #2 TEST


model2 = Sequential()
model2.add(Conv2D(200, (3, 3), activation='relu', input_shape=X.shape[1:]))
model2.add(MaxPooling2D((5, 2)))

model2.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model2.add(MaxPooling2D((3, 2)))
model2.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model2.add(MaxPooling2D((2, 2)))

model2.add(Conv2D(140,kernel_size=(3,3),activation='relu'))
model2.add(MaxPooling2D((1, 2)))


model2.add(MaxPool2D(5,5))

model2.add(Flatten())
model2.add(Dense(180, activation='relu'))
model2.add(Dense(100,activation='relu'))
model2.add(Dense(50,activation='relu'))
model2.add(Dropout(rate=0.5))
model2.add(Dense(3, activation='softmax'))

#END MODEL2 TEST
'''


'''model2 = Sequential()
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
model2.add(Dense(6, activation='softmax'))'''

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(optimizer=rmsprop(lr=0.000001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


history = model.fit(X, y, epochs=50, steps_per_epoch=2000, validation_split=0.1, callbacks =[tensorboard])

print(history.history.keys())
print(history.history.values())

test_loss, test_acc = model.evaluate(testX, testY, verbose=2)

model.save('rmsprop.h5')

print('\nTest accuracy:', test_acc)