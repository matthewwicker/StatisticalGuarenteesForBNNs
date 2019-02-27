
# coding: utf-8
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
import gc


# In[2]:


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


#N = 10  # number of data points
#D = 28 * 28 # number of features

x = tf.placeholder(tf.float32, shape = [N, 784], name = "x_placeholder")
#y_ = tf.placeholder("float", shape = [None, 10])
y_ = tf.placeholder(tf.int32, [N], name = "y_placeholder")

x_image = tf.reshape(x, [-1,28,28,1])

with tf.name_scope("model"):
    
    W_fc1 = Normal(loc=tf.zeros([784, width]), scale=tf.ones([784, width]), name="W_fc1")
    b_fc1 = Normal(loc=tf.zeros([width]), scale=tf.ones([width]), name="b_fc1")
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    
    W_fc2 = Normal(loc=tf.zeros([width, 10]), scale=tf.ones([width, 10]), name="W_fc2")
    b_fc2 = Normal(loc=tf.zeros([10]), scale=tf.ones([10]), name="b_fc2")

    #y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y = Categorical(tf.matmul(h_fc1, W_fc2) + b_fc2)


# In[5]:


# number of samples 
# we set it to 20 because of the memory constrain in the GPU.
# My GPU can take upto about 200 samples at once. 

T = 200
# INFERENCE
with tf.name_scope("posterior"):
    qW_fc1 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,784,width])))
    qb_fc1 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,width])))

    qW_fc2 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,width, 10])))
    qb_fc2 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,10])))


# In[6]:


#X_batch , Y_batch = mnist.train.next_batch(N)
#Y_batch = np.argmax(Y_batch, axis = 1)

inference = ed.HMC({W_fc1: qW_fc1, b_fc1: qb_fc1, W_fc2: qW_fc2, b_fc2: qb_fc2 }, data={y: y_})
inference.initialize(step_size=0.01, n_steps=10)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# In[7]:


for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch})
    inference.print_progress(info_dict_hmc)


# In[8]:


def test_using_last_sample(x_test, y_test):
    x_image = tf.reshape(x_test, [-1,28*28])
    #y_test = np.argmax(y_test, 1).astype("int32")
    W_fc1 = qW_fc1.eval() #qW_fc1.params[-2]
    b_fc1 = qb_fc1.eval() #qb_fc1.params[-2]
    h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)

    W_fc2 = qW_fc2.eval() #.params[-2]
    b_fc2 = qb_fc2.eval() #.params[-2]

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    y_pred = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(y_pred , y_test )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float") )
    return accuracy

X_test = mnist.test.images
Y_test = mnist.test.labels
Y_test = np.argmax(Y_test,axis=1)
accuracy = test_using_last_sample(X_test ,Y_test)
test_res = accuracy.eval()
print test_res


# In[15]:


#=============================================
# Iteration 2-n
#=============================================
#WHOOPS, WHAT DO YOU MEAN THIS IS A SYNTAX ERROR?
import gc
import six
from tqdm import tqdm
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
    X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch})
    inference.print_progress(info_dict_hmc)


# In[ ]:


EPOCHS = 3
for i in range(EPOCHS):
    assign_ops = []

    for z, qz in six.iteritems(inference.latent_vars):
        variable = qz.get_variables()[0]
        assign_ops.append(tf.scatter_update(variable, 0 , qz.params[-1]))

    tf.global_variables_initializer()
    sess.run(assign_ops)

    assign_op = inference.t.assign(0)
    sess.run(assign_op)

    for _ in range(inference.n_iter):
        X_batch, Y_batch = mnist.train.next_batch(N)
        # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
        Y_batch = np.argmax(Y_batch,axis=1)
        info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch})
        inference.print_progress(info_dict_hmc)
        
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    Y_test = np.argmax(Y_test,axis=1)
    accuracy = test_using_last_sample(X_test ,Y_test)
    test_res = accuracy.eval()
    print test_res


# In[19]:


# Sample weights and save them to a directory
#train to convergence before this
import os
if not os.path.exists("SampledModels"):
    os.makedirs("SampledModels")
from tqdm import trange
for _ in trange(400):
    np.savez_compressed("SampledModels/sample_weights_%s"%(_), [qW_fc1.eval(), 
                                                  qb_fc1.eval(), 
                                                  qW_fc2.eval(),
                                                  qb_fc2.eval()], 
                        ['wfc1', 'bfc1', 'w', 'b'])




