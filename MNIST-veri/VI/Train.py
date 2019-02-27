
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed
import pandas as pd
from tqdm import tqdm
from tqdm import trange
import os
import cv2
# In[2]:
if not os.path.exists('SampledModels'):
	os.makedirs('SampledModels')

# Use the TensorFlow method to download and/or load the data.
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True) 

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

print X_train.shape
print Y_train.shape


# In[3]:


ed.set_seed(980297)
N = 256   # number of images in a minibatch.
D = 14*14   # number of features.
K = 1    # number of classes.

width = 512


# In[4]:


# Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
x = tf.placeholder(tf.float32, [None, D])
# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
W_fc1 = Normal(loc=tf.zeros([D, width]), scale=tf.ones([D, width]))
b_fc1 = Normal(loc=tf.zeros(width), scale=tf.ones(width))
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

w = Normal(loc=tf.zeros([width, K]), scale=tf.ones([width, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
#h = tf.nn.relu(tf.matmul(h_fc1, w) + b)
# Categorical likelihood for classication.
y = Categorical(tf.matmul(h_fc1, w)+b)
#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_train, logits=y)
#cost = tf.reduce_mean(cross_entropy)

# In[5]:


qW_fc1 = Normal(loc=tf.Variable(tf.random_normal([D, width])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, width])))) 
qb_fc1 = Normal(loc=tf.Variable(tf.random_normal([width])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([width]))))

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw = Normal(loc=tf.Variable(tf.random_normal([width, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([width, K])))) 
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))


# In[6]:


# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])
# Define the VI inference technique, ie. minimise the KL divergence between q and p.
# SGHMC - for MCMC
# KLqp - for VI
inference = ed.KLqp({W_fc1: qW_fc1, qb_fc1: b_fc1, w: qw, b: qb}, data={y:y_ph})
# Initialse the infernce variables
inference.initialize(n_iter=5000, n_print=200, scale={y: float(len(Y_test)) / N})


# In[8]:


# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()


# In[9]:
def fetch_batch(size, place):
        return X_train[place:place+size], Y_train[place:place+size]

place = 0
for _ in range(inference.n_iter):
    X_batch, Y_batch = fetch_batch(N, place) #mnist.train.next_batch(N)
    if(place+N+10 >= len(X_train)):
                place = 0
    #Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)


# In[10]:

import h5py

def test_using_last_sample(x_test, y_test):
    x_image = tf.reshape(x_test, [-1,14*14])
    #y_test = np.argmax(y_test, 1).astype("int32")
    W_fc1 = qW_fc1.eval() #qW_fc1.params[-2]
    b_fc1 = qb_fc1.eval() #qb_fc1.params[-2]
    h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)

    W_fc2 = qw.eval() #.params[-2]
    b_fc2 = qb.eval() #.params[-2]

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    y_pred = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(y_pred , y_test )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float") )
    return accuracy

#X_test = mnist.test.images
#Y_test = mnist.test.labels
#Y_test = np.argmax(Y_test,axis=1)
accuracy = test_using_last_sample(X_test,Y_test)
test_res = accuracy.eval()
print "Here is our test set accuracy:",  test_res
# In[13]:


import gc
gc.collect()
n_samples = 400
for _ in trange(n_samples):
    #print _
    wfc1_samp = qW_fc1.sample()
    bfc1_samp = qb_fc1.sample()
    
    w_samp = qw.sample()
    b_samp = qb.sample()
    # We are going to save these values into a file now
    np.savez_compressed("SampledModels/sample_weights_%s"%(_), [wfc1_samp.eval(), 
                                                  bfc1_samp.eval(), 
                                                  w_samp.eval(),
                                                  b_samp.eval()], 
                        ['wfc1', 'bfc1', 'w', 'b'])

