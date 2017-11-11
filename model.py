import os
import csv
from keras.models import load_model
from pathlib import Path
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.pooling import MaxPooling2D
from random import shuffle

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

ch, row, col = 3, 160, 320  # Trimmed image format
correction = 0.1

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * -1.0)
                images.append(cv2.imread('./IMG/'+batch_sample[1].split('/')[-1]))
                angles.append(center_angle + correction)
                images.append(cv2.imread('./IMG/'+batch_sample[2].split('/')[-1]))
                angles.append(center_angle - correction)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

def getModel(model_name) :
    my_model = Path(model_name)
    if my_model.is_file() :
        model = load_model(my_model)
        return model
    # TODO: Build the Final Test Neural Network in Keras Here
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))

    model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Conv2D(64,3,3,activation="relu"))
    model.add(Conv2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

model = getModel('model.h5')
model.compile(loss="mse", optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('model.h5')
