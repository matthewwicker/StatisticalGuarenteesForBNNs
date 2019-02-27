
# coding: utf-8
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


# In[ ]:


def normalize_image(image):
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


# In[ ]:


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


# In[ ]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import edward as ed
from edward.models import Bernoulli, Normal, Categorical,Empirical
from edward.util import Progbar
from keras.layers import Dense
from scipy.misc import imsave
import matplotlib.pyplot as plt
from edward.util import Progbar
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import gc


# In[ ]:


ed.set_seed(980297)
N = 256   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.

number_of_filters = 25 # 5 also works # 1 works
filter_size = 3 # 1 works
width = 256


# In[ ]:


x = tf.placeholder(tf.float32, shape = [N,28,28,channels], name = "x_placeholder")
#y_ = tf.placeholder("float", shape = [None, 10])
y_ = tf.placeholder(tf.int32, [N], name = "y_placeholder")

#x_image = tf.reshape(x, [-1,28,28,1])
x_image = x

with tf.name_scope("model"):
    W_conv1 = Normal(loc=tf.ones([filter_size,filter_size,channels,number_of_filters]), 
                     scale=tf.ones([filter_size,filter_size,channels,number_of_filters])*0.01, name="conv1_W")
    b_conv1 = b_fc1 = Normal(loc=tf.zeros([number_of_filters]), scale=tf.ones([number_of_filters]), name="b_fc1")
    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='VALID') + b_conv1
    conv1 = tf.nn.relu(h_conv1)
    #conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    features = tf.contrib.layers.flatten(conv1)
    
    W_fc1 = Normal(loc=tf.zeros([16900, width]), scale=tf.ones([16900, width]), name="W_fc1")
    b_fc1 = Normal(loc=tf.zeros([width]), scale=tf.ones([width]), name="b_fc1")
    h_fc1 = tf.nn.relu(tf.matmul(features, W_fc1) + b_fc1)
    
    W_fc2 = Normal(loc=tf.zeros([width, desired_classes]), scale=tf.ones([width, desired_classes]), name="W_fc2")
    b_fc2 = Normal(loc=tf.zeros([desired_classes]), scale=tf.ones([desired_classes]), name="b_fc2")

    #y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y = Categorical(tf.matmul(h_fc1, W_fc2) + b_fc2)


# In[ ]:


# number of samples 
# we set it to 20 because of the memory constrain in the GPU.
# My GPU can take upto about 200 samples at once. 

T = 100
# INFERENCE
with tf.name_scope("posterior"):
    qW_conv1 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,filter_size,filter_size,channels,number_of_filters])))
    qb_conv1 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,number_of_filters])))
    
    qW_fc1 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,16900,width])))
    qb_fc1 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,width])))

    qW_fc2 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,width, desired_classes])))
    qb_fc2 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,desired_classes])))


# In[ ]:



inference = ed.HMC({W_conv1: qW_conv1,
                      b_conv1: qb_conv1,
                      W_fc1: qW_fc1,
                      b_fc1: qb_fc1,
                      W_fc2: qW_fc2,
                      b_fc2: qb_fc2 }, data={y: y_})

inference.initialize(step_size=0.001, n_steps=3)
#inference.initialize(step_size=0.0005, n_steps=5)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# In[ ]:


counter = 0


# In[ ]:


def get_batch(n, counter):
    return X_train[counter:counter+n], y_train[counter:counter+n]

for _ in range(inference.n_iter):
    X_batch, Y_batch = get_batch(N,counter)
    counter+=N
    if(counter+N+1 >= len(X_train)):
        counter = 0
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch})
    inference.print_progress(info_dict_hmc)


# In[ ]:


def test_using_last_sample(x_test, y_test):
    x_image = tf.reshape(x_test, [-1,28,28,channels])
    W_conv1 = qW_conv1.eval()
    b_conv1 = qb_conv1.eval()
    W_fc1 = qW_fc1.eval() #qW_fc1.params[-2]
    b_fc1 = qb_fc1.eval() #qb_fc1.params[-2]
    W_fc2 = qW_fc2.eval() #.params[-2]
    b_fc2 = qb_fc2.eval() #.params[-2]
    
    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='VALID') + b_conv1
    conv1 = tf.nn.relu(h_conv1)
    #conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    features = tf.contrib.layers.flatten(conv1)
    
    h_fc1 = tf.nn.relu(tf.matmul(features, W_fc1) + b_fc1)

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    
    y_pred = tf.argmax(y_conv, 1)
    
    correct_prediction = tf.equal(y_pred , y_test )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float") )
    return accuracy

X_train = X_train.astype('float32')
Y_train = np.argmax(y_train,axis=1)
accuracy = test_using_last_sample(X_train ,Y_train)
test_res = accuracy.eval()
print test_res

X_test = X_test.astype('float32')
Y_test = np.argmax(y_test,axis=1)
accuracy = test_using_last_sample(X_test ,Y_test)
test_res = accuracy.eval()
print test_res

X_valid = X_valid.astype('float32')
Y_valid = np.argmax(y_valid,axis=1)
accuracy = test_using_last_sample(X_valid ,Y_valid)
test_res = accuracy.eval()
print test_res


# In[ ]:


import gc
import six
from tqdm import tqdm

EPOCHS = 1
for i in range(EPOCHS):
	assign_ops = []

	gc.collect()

	for z, qz in six.iteritems(inference.latent_vars):
	    variable = qz.get_variables()[0]
	    assign_ops.append(tf.scatter_update(variable, 0 , qz.params[-1]))

	tf.global_variables_initializer()
	sess.run(assign_ops)

	assign_op = inference.t.assign(0)
	sess.run(assign_op)

	gc.collect()

	for _ in range(inference.n_iter):
	    X_batch, Y_batch = get_batch(N,counter)
	    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
	    Y_batch = np.argmax(Y_batch,axis=1)
	    info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch})
	    inference.print_progress(info_dict_hmc)

	test_set_results = []
	valid_set_results = []
	for i in range(20):
	    accuracy = test_using_last_sample(X_test ,Y_test)
	    test_res = accuracy.eval()
	    test_set_results.append(test_res)
	    
	    accuracy = test_using_last_sample(X_valid ,Y_valid)
	    test_res = accuracy.eval()
	    valid_set_results.append(test_res)
	    
	print(np.average(test_set_results))

	print(np.average(valid_set_results))


# In[ ]:



import os
if not os.path.exists("SampledModels"):
    os.makedirs("SampledModels")
from tqdm import trange
for _ in trange(400):
    np.savez_compressed("SampledModels/sample_weights_%s"%(_), [qW_conv1.eval(),
                                                    qb_conv1.eval(),
                                                    qW_fc1.eval(), 
                                                    qb_fc1.eval(), 
                                                    qW_fc2.eval(),
                                                    qb_fc2.eval()], 
                        ['convw1', 'convb1', 'wfc1', 'bfc1', 'w', 'b'])
                        

