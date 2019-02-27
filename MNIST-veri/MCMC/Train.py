
# coding: utf-8

# In[1]:


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
import cv2

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
D = 14*14   # number of features.
K = 1    # number of classes.

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

width = 512

# In[4]:
x = tf.placeholder(tf.float32, shape = [N, 14*14], name = "x_placeholder")
#y_ = tf.placeholder("float", shape = [None, 10])
y_ = tf.placeholder(tf.int32, [N], name = "y_placeholder")

#x_image = tf.reshape(x, [-1,28,28,1])

with tf.name_scope("model"):
    
    W_fc1 = Normal(loc=tf.zeros([D, width]), scale=tf.ones([D, width]), name="W_fc1")
    b_fc1 = Normal(loc=tf.zeros([width]), scale=tf.ones([width]), name="b_fc1")
    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    
    W_fc2 = Normal(loc=tf.zeros([width, K]), scale=tf.ones([width, K]), name="W_fc2")
    b_fc2 = Normal(loc=tf.zeros([K]), scale=tf.ones([K]), name="b_fc2")

    #y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y = Categorical(tf.matmul(h_fc1, W_fc2) + b_fc2)
    #y = tf.nn.sigmoid_cross_entropy_with_logits(tf.matmul(h_fc1,W_fc2)+b_fc2)
    #tf.nn.sigmoid_cross_entropy_with_logits(y)
# In[5]:


# number of samples 
# we set it to 20 because of the memory constrain in the GPU.
# My GPU can take upto about 200 samples at once. 

T = 200
# INFERENCE
with tf.name_scope("posterior"):
    qW_fc1 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,D,width])))
    qb_fc1 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,width])))

    qW_fc2 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,width,K])))
    qb_fc2 = Empirical(params = tf.Variable(1/100 *tf.random_normal([T,K])))


# In[6]:


#X_batch , Y_batch = mnist.train.next_batch(N)
#Y_batch = np.argmax(Y_batch, axis = 1)

inference = ed.HMC({W_fc1: qW_fc1, b_fc1: qb_fc1, W_fc2: qW_fc2, b_fc2: qb_fc2 }, data={y: y_})
inference.initialize(step_size=0.01, n_steps=10)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


def fetch_batch(size, place):
	return X_train[place:place+size], Y_train[place:place+size]

# In[7]:

place = 0
for _ in range(inference.n_iter):
    X_batch, Y_batch = fetch_batch(N, place) #mnist.train.next_batch(N)
    if(place+N+10 >= len(X_train)):
		place = 0
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    #Y_batch = np.argmax(Y_batch,axis=1)
    info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch})
    inference.print_progress(info_dict_hmc)


# In[8]:


def test_using_last_sample(x_test, y_test):
    x_image = tf.reshape(x_test, [-1,14*14])
    #y_test = np.argmax(y_test, 1).astype("int32")
    W_fc1 = qW_fc1.eval() #qW_fc1.params[-2]
    b_fc1 = qb_fc1.eval() #qb_fc1.params[-2]
    h_fc1 = tf.nn.relu(tf.matmul(x_image, W_fc1) + b_fc1)

    W_fc2 = qW_fc2.eval() #.params[-2]
    b_fc2 = qb_fc2.eval() #.params[-2]

    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    print len(y_conv.eval())
    print sum(y_conv.eval())
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




