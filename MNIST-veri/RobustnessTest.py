import warnings, logging, sys
import cv2
import gc
import os
import time
import shutil
import random
import argparse

import pickle
import numpy as np
import pandas as pd
from scipy.misc import imsave
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import edward as ed
from edward.util import Progbar
from edward.models import Bernoulli, Normal, Categorical, Empirical

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import InputLayer, Reshape, Flatten
from keras.utils import np_utils
from keras.models import model_from_json
from keras.datasets import mnist
from keras import regularizers

from Properties import *

from utils import keep_only_spec_labels # Hey as long as this works :)


warnings.filterwarnings('ignore')
logging.disable(sys.maxsize)
parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--eps")
parser.add_argument("--mode")
parser.add_argument("--path")
parser.add_argument("--delt")

args = parser.parse_args()
image_num = int(args.imnum)
max_eps = float(args.eps)
mode = int(args.mode)
model_path = str(args.path)
max_delt = float(args.delt)

weights_path = model_path + "SampledModels"

channels = 1

(_, _), (X_test, y_test) = mnist.load_data()
X_test, _ = keep_only_spec_labels(X_test/255.,y_test,1,7)

N = 256   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.
width = 512


import math
from statsmodels.stats.proportion import proportion_confint

# also known as the chernoff bound
def okamoto_bound(epsilon, delta):
    return (-1*.5) * math.log(float(delta)/2) * (1.0/(epsilon**2))

# This is h_a in the paper
def absolute_massart_halting(succ, trials, I, epsilon):
    gamma = float(succ)/trials
    if(I[0] < 0.5 and I[1] > 0.5):
        return -1
    elif(I[1] < 0.5):
        val = I[1]
        h = (9/2.0)*(((3*val + epsilon)*(3*(1-val)-epsilon))**(-1))
        return math.ceil((h*(epsilon**2))**(-1) * math.log((delta - alpha)**(-1)))
    elif(I[0] >= 0.5):
        val = I[0]
        h = (9/2.0)*(((3*(1-val) + epsilon)*((3*val)+epsilon))**(-1))
        return math.ceil((h*(epsilon**2))**(-1) * math.log((delta - alpha)**(-1)))
    

"""
For now this algorithm is set up to solve problem formulation one
from our ICML2019 paper.

With m_delta = <r> we check the <r>-robustness of the model with respect
    to the property specified (example properties above)
with m_detla = -1 we check the probabalistic safety of the model 

@param epsilon: the permitted error in our estimated 'safety' variable.
@param delta: the permitted probability that our estimated 'safety' variable is incorrect
@param alpha: significance value used in exact clopper-pearson interval estimate
@param testproperty: method to verify property of interest(see documentation for format)
@param inp: the test input to check
@param out: the ground truth (optional)
@param m_delta: safety/robustness radius to check (see above).
"""
def sequential_massart(epsilon, delta, alpha, networks, testproperty, inp,
                       m_delta, out=None, max_k=1, attacking=False, respath=''):
    atk_locs = []
    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    #print("Maximum bound = %s"%(chernoff_bound))
    successes, iterations, misses = 0.0, 0.0, 0.0
    halting_bound = chernoff_bound
    I = [0,1]
    print "Maximum halting bound: ", halting_bound
    while(iterations <= halting_bound):
        #clear_output(wait=True)
        if(iterations > 0):
            print("Working on iteration: %s \t Bound: %s \t Param: %s"%(iterations, halting_bound, successes/iterations))    
        try:
            model.load_weights(networks[int(iterations)])
        except:
            m = np.load(networks[int(iterations)])
            start = 1
            model.layers[start].set_weights([m['arr_0'][0],m['arr_0'][1]])
            model.layers[start+1].set_weights([m['arr_0'][2], m['arr_0'][3]])
        result = testproperty(inp, out, model, m_delta, max_k=max_k, path=respath)

        if(result == -1):
            misses += 1
            result = 0
        successes += result
        iterations += 1
        # Setting the method equal to 'beta' here gives us clopper-pearson
        # and ensures that these bounds are exact.
        lb, ub = proportion_confint(successes, iterations, method='beta')
        I = [lb, ub]
        hb = absolute_massart_halting(successes, iterations, I, epsilon)
        if(hb == -1):
            halting_bound = chernoff_bound
        else:
            halting_bound = min(hb, chernoff_bound)
    print("Exited becuase %s >= %s"%(iterations, halting_bound))
    return successes/iterations, misses/iterations


propertymodel = Sequential()
propertymodel.add(Dense(512, activation='relu', input_shape=(14*14,)))
propertymodel.add(Dense(1, activation='softmax'))
propertymodel.summary()
propertymodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


attackingmodel = Sequential()
attackingmodel.add(InputLayer(input_shape=(1,14,14)))
attackingmodel.add(Reshape((1,14*14)))
attackingmodel.add(Dense(512, activation='relu'))
attackingmodel.add(Dense(1, activation='softmax'))
attackingmodel.summary()
attackingmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


verificationmodel = Sequential()
verificationmodel.add(Flatten(input_shape = (14,14,1)))
verificationmodel.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001),
                bias_regularizer=regularizers.l2(0.001)))
#verificationmodel.add(Dropout(0.5))
verificationmodel.add(Dense(1, activation='sigmoid'))
verificationmodel.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

if(mode==0):
	verifier = FGSM_verifier
	veri_directory = "FGSM"
	model = attackingmodel
elif(mode==1):
	verifier = PGD_verifier
	veri_directory = "PGD"
	model = attackingmodel
elif(mode==2):
	verifier = translational_verifier
	if(max_eps == 0.075):
		max_eps = 1
	elif(max_eps == 0.1):
		max_eps = 2
	else:
		max_eps = 3
	veri_directory = "TRANS"
	model = attackingmodel
elif(mode==3):
	verifier = rotational_verifier
	if(max_eps == 0.075):
		max_eps = 1
	elif(max_eps == 0.1):
		max_eps = 2
	else:
		max_eps = 3
	veri_directory = "ROTA"
	model = attackingmodel
elif(mode==4):
	verifier = CWL2_verifier
	if(max_eps == 0.075):
		max_eps = 1
	elif(max_eps == 0.1):
		max_eps = 2
	else:
		max_eps = 3
	veri_directory = "CW"
	model = attackingmodel
elif(mode==5):
    verifier = DeepGO 
    if(max_eps == 0):
		max_eps = 0.25
    elif(max_eps == 1):
		max_eps = 0.35
    elif(max_eps == 2):
		max_eps = 0.45
    elif(max_eps == 3):
		max_eps = 0.55
    elif(max_eps == 4):
		max_eps = 0.65
    elif(max_eps == 5):
		max_eps = 0.75
    if(max_delt == 0):
                max_delt = 0.0001
    elif(max_delt == 1):
                max_delt = 0.0005
    elif(max_delt == 2):
                max_delt = 0.001
    elif(max_delt == 3):
                max_delt = 0.005
    elif(max_delt == 4):
                max_delt = 0.01
    elif(max_delt == 5):
                max_delt = 0.05
    veri_directory = "DeepGO"
    model = verificationmodel


from os import listdir
from os.path import isfile, join
from utils import averagePooling
# Load in the model weights from this training
weights_path = weights_path + '/'
model_weights = [weights_path + f for f in listdir(weights_path) if (isfile(join(weights_path, f)) and f[0]!='.' ) ]
print len(model_weights)

epsilon = 0.075
delta = 0.075
alpha = 0.05
x_test = X_test
m_delta = -1 #max_delt
sess = backend.get_session()

attack_locals = []
start = time.time()

inp_image = x_test[image_num]
if verifier == DeepGO:
    inp_image = averagePooling([inp_image])
    inp_image = inp_image[0]


if not os.path.exists(model_path + "Results/"):
    os.mkdir(model_path + "Results/")

if not os.path.exists(model_path + "Results/%s_Attacks/"%(veri_directory)):
    os.mkdir(model_path + "Results/%s_Attacks/"%(veri_directory))

if not os.path.exists(model_path + "Results/%s_Attacks/image_%s_eps_%s_delt_%s"%(veri_directory, image_num,max_eps,m_delta)):
    os.mkdir(model_path +"Results/%s_Attacks/image_%s_eps_%s_delt_%s"%(veri_directory, image_num,max_eps,m_delta))

respath = model_path + "Results/%s_Attacks/image_%s_eps_%s_delt_%s"%(veri_directory, image_num, max_eps, m_delta)

val, misses = sequential_massart(epsilon, delta, alpha, model_weights,
                             verifier, inp_image, m_delta, max_k=max_eps, attacking=True, respath=respath)
end = time.time()
# Need to save these
try:
     avg = sum(attack_locals)/len(attack_locals)
except:
    avg = np.zeros((28,28))
avg = np.reshape(avg, (28,28))
variation = np.zeros((28,28))
for i in attack_locals:
    variation += (avg-i)**2
try:
    variation/=N
except:
    print("Divide by zero... but continuing")

var = np.reshape(variation, (28,28))
f=open(model_path + "Results/Stats-%s.txt"%(veri_directory), "a+")
f.write("|%s - %s - %s (%s, %s)|"%(image_num,val,end-start,max_eps,m_delta))
#np.savetxt(model_path + "FGSM_Attacks/image_%s_eps_%s/original_image_val=%s.txt"%(image_num, max_eps, val), x_test[image_num])
cv2.imwrite(model_path + "Results/%s_Attacks/image_%s_eps_%s_delt_%s/original_image_val=%s.png"%(veri_directory, image_num, max_eps,m_delta, val), x_test[image_num]+0.5)

np.savetxt(model_path + "Results/%s_Attacks/image_%s_eps_%s_delt_%s/average_manip_b.txt"%(veri_directory, image_num,max_eps,m_delta), avg)

np.savetxt(model_path + "Results/%s_Attacks/image_%s_eps_%s_delt_%s/variance_b.txt"%(veri_directory,image_num,max_eps,m_delta), var)
