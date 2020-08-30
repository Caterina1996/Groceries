# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:27:24 2020

@author: Caterina
"""

import numpy as np
import tensorflow as tf

import sklearn.metrics as sk_metrics
from matplotlib import pyplot as plt

import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dropout, Flatten, Dense, Conv2D, BatchNormalization, MaxPooling2D
from keras import applications

from sklearn.metrics import confusion_matrix
import itertools
from keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input

import glob
import ntpath as nt
import os
from PIL import Image

import operator
import shutil

from keras.preprocessing import image


PATH_GROCERIES = 'C:\\Users\\Caterina\\Documents\\GitHub\\TFM\\Groceries\\images\\'
PATH_GROCERIES_TRAIN = 'C:\\Users\\Caterina\\Documents\\GitHub\\TFM\\Groceries\\ORDENAT\\Train\\'
PATH_GROCERIES_TEST = 'C:\\Users\\Caterina\\Documents\\GitHub\\TFM\\Groceries\\ORDENAT\\Test\\'


classes_directories=[x[0] for x in os.walk(PATH_GROCERIES)]

classes=os.listdir(PATH_GROCERIES)

train_split=0.85
tidied=True

img_width,img_height=224,224
nb_train_samples = 0
nb_validation_samples = 0

epochs = 50
batch_size = 32

def tidy_data(split_ratio,tidied=True):
    
    nb_train_samples = 0
    nb_test_samples = 0 
    
    if tidied==False:
         
         
        for classe in classes:
           os.makedirs(PATH_GROCERIES_TEST+classe)
           os.makedirs(PATH_GROCERIES_TRAIN+classe)
           
           path=PATH_GROCERIES+classe
           
           allFileNames = os.listdir(path)
           np.random.shuffle(allFileNames)
           train_FileNames, test_FileNames= np.split(np.array(allFileNames),
                                                           [int(len(allFileNames)* split_ratio)])
           
           
           
           train_FileNames = [path + '\\'+name for name in train_FileNames.tolist()]
           
          
           test_FileNames = [path + '\\' + name for name in test_FileNames.tolist()]
           
           print('Total images: ', len(allFileNames))
           
           print('Training: ', len(train_FileNames))
           nb_train_samples=nb_train_samples+len(train_FileNames)
           
           print('Testing: ', len(test_FileNames))
           nb_validation_samples=nb_validation_samples+len(test_FileNames)
           # Copy-pasting images
           for name in train_FileNames:
               shutil.copy(name, PATH_GROCERIES_TRAIN + classe)
               
           
           for name in test_FileNames:
               shutil.copy(name, PATH_GROCERIES_TEST + classe)
               
    else:
        for classe in classes:
            nb_train_samples=nb_train_samples+len(os.listdir(PATH_GROCERIES_TRAIN+"\\"+classe))
            
            nb_test_samples=nb_test_samples+len(os.listdir(PATH_GROCERIES_TEST+"\\"+classe))
               
             
    return nb_train_samples,nb_test_samples


#Ordenar el dataset:             
nb_train_samples,nb_test_samples=tidy_data(train_split,tidied)

#Cargar los datos:

datagen = ImageDataGenerator(rescale=1. / 255)

train_set = datagen.flow_from_directory(
    'ORDENAT\Train',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

test_set = datagen.flow_from_directory(
    'ORDENAT\Test',
    target_size=(img_width,img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

test_labels=tf.one_hot(test_set.classes,max(test_set.classes+1))
train_labels=tf.one_hot(train_set.classes,max(train_set.classes+1))

#Definimos el modelo y obtenemos bottleneck features:

resnet_model=tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet")

#Check that the resnet model works
img = image.load_img('elephant.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = resnet_model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

resnet_model.summary()

#Charge the model now without the last layer
resnet_model=tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet")

resnet_model.summary()

bottleneck_features_train=resnet_model.predict_generator(
    train_set, len(train_set.filenames) // batch_size +1)

#print("bottleneck_features_train ",bottleneck_features_train)
np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)

bottleneck_features_validation = resnet_model.predict_generator(
    test_set, nb_validation_samples // batch_size+1)

#print("bottleneck_features_val ",bottleneck_features_validation)
 
np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)


train_data = np.load(open('bottleneck_features_train.npy', mode="rb"))

validation_data = np.load(open('bottleneck_features_validation.npy', mode="rb"))



feature_model=Sequential()
feature_model.add(Dense(25, activation="sigmoid"))
feature_model.compile(loss="categorical_crossentropy", 
                      optimizer="adam", metrics=["accuracy"])

feature_model.fit(train_data, train_labels,
          epochs=50,
          validation_data=(validation_data, test_labels))


feature_model.fit_generator(train_set,
                         steps_per_epoch = nb_train_samples // batch_size,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = nb_test_samples // batch_size)


Classifier.save('prueba_Groceries.h5')


