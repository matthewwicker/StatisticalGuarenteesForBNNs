
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
import cv2

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True) 

sess = tf.InteractiveSession()

batch_size = 128
num_classes = 1

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
                Y_test.append(classes.index(np.argmax(y_test[i])))
                x = np.reshape(x_test[i], (28,28))
                x = cv2.resize(x, (14,14))
                X_test.append(x)

X_train = np.reshape(X_train,(-1,14*14))
X_test = np.reshape(X_test,(-1,14*14))
Y_train = np.asarray(Y_train)*0.9999
Y_test = np.asarray(Y_test)*0.9999


#val = 0 
#for i in range(14):
#	print X_test[0][val:val+14]
#	val+=14

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(14*14,)))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


from os import listdir
from os.path import isfile, join
mypath = 'SampledModels/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
scores = []

input = np.asarray(X_test[2]).reshape((1,14*14))
print Y_test[2]
#Y_train = 0.9999-Y_train
#Y_test = 0.9999-Y_test
print "Class break down of Y_test:"
print "Number of sevens: ", sum(Y_test)
print "Number of ones: ", len(Y_test) - sum(Y_test)


def test_using_last_sample(x_test, y_test, W_fc1, b_fc1, W_fc2, b_fc2):
    x_image = tf.reshape(x_test, [-1,14*14])
    #y_test = np.argmax(y_test, 1).astype("int32")
    #W_fc1 = qW_fc1.eval() #qW_fc1.params[-2]
    #b_fc1 = qb_fc1.eval() #qb_fc1.params[-2]
    h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)

    #W_fc2 = qW_fc2.eval() #.params[-2]
    #b_fc2 = qb_fc2.eval() #.params[-2]

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    print len(y_conv.eval())
    print sum(y_conv.eval())
    y_pred = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(y_pred , y_test )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float") )
    #return accuracy

    test_res = accuracy.eval()
    print "Here is our test set accuracy:",  test_res
    return test_res

from tqdm import tqdm
for weight in tqdm(onlyfiles):
	try:
		model.load_weights(mypath + weight)
        except:
            m = np.load(mypath + weight)
            start = 0
            model.layers[start].set_weights([m['arr_0'][0],m['arr_0'][1]])
            model.layers[start+1].set_weights([m['arr_0'][2], m['arr_0'][3]])

	score = model.evaluate(X_test,Y_test, verbose=0)
	score = test_using_last_sample(X_test, Y_test, m['arr_0'][0], m['arr_0'][1], m['arr_0'][2], m['arr_0'][3])
	#print score
	scores.append(score)
#np.savetxt('accuracies.txt', scores)
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

