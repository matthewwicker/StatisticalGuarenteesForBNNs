
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

# In[2]:
if not os.path.exists('SampledModels'):
	os.makedirs('SampledModels')

# Use the TensorFlow method to download and/or load the data.
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True) 
X = mnist.train.images
print X.shape
X = mnist.test.images
print X.shape


# In[3]:


ed.set_seed(980297)
N = 256   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.

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
inference.initialize(n_iter=27500, n_print=200, scale={y: float(mnist.train.num_examples) / N})


# In[8]:


# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()


# In[9]:


# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)


# In[10]:


# Load the test images.
X_test = mnist.test.images
# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.argmax(mnist.test.labels,axis=1)


# In[11]:


# Generate samples the posterior and store them.
import h5py

n_samples = 5
prob_lst = []
samples = []
wfc1_samples = []
bfc1_samples = []
w_samples = []
b_samples = []
for _ in trange(n_samples):
    wfc1_samp = qW_fc1.sample()
    bfc1_samp = qb_fc1.sample()
    wfc1_samples.append(wfc1_samp)
    bfc1_samples.append(bfc1_samp)
    
    w_samp = qw.sample()
    b_samp = qb.sample()
    w_samples.append(w_samp)
    b_samples.append(b_samp)
    # We are going to save these values into a file now
    #np.savez_compressed("SampledModels/sample_weights_%s"%(_), [wfc1_samp.eval(), 
    #                                              bfc1_samp.eval(), 
    #                                              w_samp.eval(),
    #                                              b_samp.eval()], 
    #                    ['wfc1', 'bfc1', 'w', 'b'])
    # Also compue the probabiliy of each class for each (w,b) sample.
    h_fc1 = tf.nn.relu(tf.matmul(X_test, wfc1_samp) + bfc1_samp)
    prob = tf.nn.softmax(tf.matmul(h_fc1, w_samp) + b_samp)
    prob_lst.append(prob.eval())
    sample = tf.concat([tf.reshape(w_samp,[-1]),b_samp],0)
    samples.append(sample.eval())
    


# In[12]:


# Compute the accuracy of the model. 
# For each sample we compute the predicted class and compare with the test labels.
# Predicted class is defined as the one which as maximum proability.
# We perform this test for each (w,b) in the posterior giving us a set of accuracies
# Finally we make a histogram of accuracies for the test data.
accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)
    acc = (y_trn_prd == Y_test).mean()*100
    accy_test.append(acc)

print np.mean(accy_test)


# In[13]:


import gc
gc.collect()
n_samples = 400
for _ in trange(5, n_samples):
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

