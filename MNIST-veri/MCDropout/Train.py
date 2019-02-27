
# coding: utf-8
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import InputLayer
from keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True) 

batch_size = 128
num_classes = 1
epochs = 10

x_train = mnist.train.images
x_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

Y_train = []
X_train = []
classes = [1,7]
for i in range(len(x_train)):
	if(np.argmax(y_train[i]) in classes):
		Y_train.append(classes.index(np.argmax(y_train[i])))
		x = np.reshape(x_train[i], (28,28))
                x = cv2.resize(x, (14,14))
		X_train.append(x)


Y_test = []
X_test = []
for i in range(len(x_test)):
        if(np.argmax(y_test[i]) in classes):
		x = np.reshape(x_test[i], (28,28))
                x = cv2.resize(x, (14,14))
                Y_test.append(classes.index(np.argmax(y_test[i])))
                X_test.append(cv2.resize(x, (14,14)))

print Y_test
X_train = np.reshape(X_train,(-1,14,14,1))
X_test = np.reshape(X_test,(-1,14,14,1))
Y_test = np.asarray(Y_test)#*0.9999
Y_train = np.asarray(Y_train)#*0.9999

model = Sequential()
model.add(InputLayer(input_shape=(14,14,1)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['acc'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[3]:


from keras.models import model_from_json
model_json = model.to_json()
with open("mnist-mlp.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("mnist-mlp.h5")
print("Saved model to disk")

