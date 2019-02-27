
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json

import pickle
import numpy as np

import matplotlib.pyplot as plt
import cv2
import pandas as pd

import os
import shutil

training_file = '../data/train.p'
validation_file= '../data/validate.p'
testing_file = '../data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape


n_classes = len(np.unique(y_train))
channels = 3

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# In[2]:


def normalize_image(image):
    #return -0.5 + (image*1.0)/(255)
    return (image*1.0)/(255)
def Gray_image(image):
    if(channels == 1):
        return np.resize(cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)[:,:,0],(28,28,1))
    return image

def preprocess(image):
    img= [] 
    for i in image:
        img.append(normalize_image(Gray_image(i)))
    img = np.array(img)        
    return img

X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test  = preprocess(X_test)

# 6. Preprocess class labels
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
y_valid = np_utils.to_categorical(y_valid, n_classes)

print "imageshape after grayscale",X_train[0].shape


# In[3]:


# We are going to be limiting the classes so we do that here

desired_classes = 10
numeric = np.argmax(y_train,axis=1)
counts = np.bincount(numeric)
classes_to_use = counts.argsort()[-desired_classes:][::-1]
print classes_to_use
classes_to_use  = [2,25] # We preselect classes so that they are the most different
desired_classes = 2  # We reset the class counter

def filter_by_class(x_data, y_data):
    X = []
    Y = []
    for i in range(len(x_data)):
        if(np.argmax(y_data[i]) in classes_to_use):
            X.append(x_data[i])
            Y.append(np.where(classes_to_use == np.argmax(y_data[i])))
    X = np.asarray(X)
    Y = np.asarray(np_utils.to_categorical(Y))
    return X, Y

X_train, y_train = filter_by_class(X_train, y_train)
X_valid, y_valid = filter_by_class(X_valid, y_valid)
X_test, y_test = filter_by_class(X_test, y_test)

image_shape = X_train[0].shape
print "X_train",np.shape(X_train)
print "y_train",np.shape(y_train)
print("Image data shape =", image_shape)
print("Number of classes =", desired_classes)


# In[4]:


model = Sequential()
model.add(Conv2D(25, (3, 3), activation='relu', input_shape=(28,28,channels)))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(desired_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#model.fit(X_train, y_train, 
#          batch_size=32, epochs=10, verbose=1)


from tqdm import tqdm
from os import listdir
from os.path import isfile, join
mypath = 'SampledModels/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
scores = []
from tqdm import trange
for i in trange(200):
	weight = onlyfiles[i]
        try:
            	model.load_weights(mypath + weight)
        except:
            	m = np.load(mypath+weight)
            	start = 0
            	model.layers[start].set_weights([m['arr_0'][0],m['arr_0'][1]])
            	model.layers[start+2].set_weights([m['arr_0'][2], m['arr_0'][3]])
            	model.layers[start+3].set_weights([m['arr_0'][4], m['arr_0'][5]])

        score = model.evaluate(X_test,y_test, verbose=0)
	#if(np.argmax(score) == np.argmax(y_test[image])):
        #curr_acc.append(np.argmax(score) == np.argmax(y_test[image]))
        print score
	scores.append(score[1])
np.savetxt('accuracies-test.txt', scores)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#score = model.evaluate(X_test, y_test, verbose=0)
#print 'score',score


# In[5]:

"""
# serialize model to JSON
from keras.models import model_from_json
model_json = model.to_json()
with open("gtsrb-model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("gtsrb-model.h5")
print("Saved model to disk")
"""
