import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


from sklearn.datasets import load_files
import matplotlib.image as mpimg
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

import os
import glob

train_files = glob.glob("./fruits/food224/Training/*/*.jpg")
labels = [i.split('/')[5] for i in train_files]

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

for i , idx in enumerate(train_files):
    train_files[i] = img_to_array(load_img(train_files[i]))

train_files = np.array(train_files).astype('float32')/255


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=2, padding='same', activation='relu', 
                                 kernel_initializer=tf.keras.initializers.Orthogonal(),
                                 input_shape=(IMG_HW, IMG_HW, 3)))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2014, activation='relu'))
model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(0.0015),
              metrics=['accuracy'])

history = model.fit(train_files , encoded_labels,
        batch_size = 32,
        epochs=30,
#         validation_data=(X_valid, y_vaild),
        verbose=2, shuffle=True)

print('Saving model')
model.save('./foodai.weigths')
print('Model Saved')
