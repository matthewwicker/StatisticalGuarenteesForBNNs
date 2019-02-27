
# coding: utf-8

# In[1]:

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 

batch_size = 128
num_classes = 10
epochs = 10

x_train = mnist.train.images
x_test = mnist.test.images
y_train = mnist.train.labels
y_test = mnist.test.labels

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


from os import listdir
from os.path import isfile, join
mypath = 'SampledModels/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
scores = []
from tqdm import tqdm
for weight in tqdm(onlyfiles):
	try:
		model.load_weights(mypath + weight)
        except:
            m = np.load(mypath + weight)
            start = 2
            model.layers[start].set_weights([m['arr_0'][0],m['arr_0'][1]])
            model.layers[start+1].set_weights([m['arr_0'][2], m['arr_0'][3]])

	score = model.evaluate(x_test, y_test, verbose=0)
	scores.append(score[1])
np.savetxt('accuracies.txt', scores)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[3]:


#from keras.models import model_from_json
#model_json = model.to_json()
#with open("mnist-mlp.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("mnist-mlp.h5")
#print("Saved model to disk")

