from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope, to_categorical
from keras.initializers import glorot_uniform
import random
import cv2
import numpy as np
import os


# DATADIR = "C:/Users/Kowalski/PycharmProjects/ML/Bite-detection/"
DATADIR = "./"

CATEGORIES = ["overbitex", "underbitex"]
data = []
for category_num in range(2):
    category = CATEGORIES[category_num]
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (256, 256))
        data.append([new_array, category_num])

random.shuffle(data)

X = []
y = []
for features, label in data:
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

trainX = np.delete(X, np.s_[0:10], axis=0)
trainY = np.delete(y, np.s_[0:10], axis=0)
testX_img = X[0:20]
testY = y[0:20]

X = X.reshape((-1, 256, 256, 1))
testX = testX_img.reshape((-1, 256, 256, 1))

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    existing_model = load_model('./finalmodel.h5')
    # Show the model architecture
existing_model.summary()
loss, acc = existing_model.evaluate(X,  (y), verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

pred = existing_model.predict_classes(testX)
# show the inputs and predicted outputs
for i in range(len(testX)):
	print("X=%s, Predicted=%s" % (testY[i], pred[i]))
print(pred, y[0])


plt.figure(figsize=(20,20))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(testX_img[i], cmap=plt.cm.binary)
    plt.xlabel("Predicted: "+str(CATEGORIES[pred[i][0]]) + "\nReal: " + str(CATEGORIES[testY[i]]))
plt.subplots_adjust(hspace=.5)
plt.show()
