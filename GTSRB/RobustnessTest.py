import warnings, logging, sys
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
import pickle
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import shutil
import numpy as np

from Properties import *
from LoadData import load

# ============================= LOADING DATA ==========================
warnings.filterwarnings('ignore')
logging.disable(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument("--imnum")
parser.add_argument("--sev")
parser.add_argument("--mode")
parser.add_argument("--path")

args = parser.parse_args()
image_num = int(args.imnum)
sev = float(args.sev)
mode = int(args.mode)
model_path = str(args.path)
weights_path = model_path + "SampledModels"


training_file = './data/train.p'
validation_file= './data/validate.p'
testing_file = './data/test.p'
desired_classes = 2
X_train, y_train, X_test, y_test = load(training_file, validation_file, testing_file)
# ============================= LOADING DATA ==========================

width = 256

if(mode==0):
	verifier = FGSM_verifier
	if(sev == 0):
                max_eps = 0.05
        elif(sev == 1):
                max_eps = 0.10
	elif(sev == 2):
                max_eps = 0.25
        elif(sev == 3):
                max_eps = 0.50
        else:
                max_eps = 0.75
	veri_directory = "FGSM"
elif(mode==1):
	verifier = PGD_verifier
	if(sev == 0):
                max_eps = 0.05
        elif(sev == 1):
                max_eps = 0.10
        elif(sev == 2):
                max_eps = 0.25
        elif(sev == 3):
                max_eps = 0.50
        else:
                max_eps = 0.75

	veri_directory = "PGD"
elif(mode==2):
	verifier = translational_verifier
	if(sev == 0):
		max_eps = 1
	elif(sev == 1):
		max_eps = 2
	elif(sev == 2):
		max_eps = 3
	elif(sev == 3):
		max_eps = 4
	else:
		max_eps = 5
	veri_directory = "TRANS"
elif(mode==3):
	verifier = rotational_verifier
	if(sev == 0):
		max_eps = 1
	elif(sev == 1):
		max_eps = 2
	elif(sev == 2):
		max_eps = 3
	elif(sev == 3):
		max_eps = 4
	else:
		max_eps = 5
	veri_directory = "ROTA"
elif(mode==4):
	verifier = CWL2_verifier
	if(sev == 1):
                max_eps = 0.05
        elif(sev == 2):
                max_eps = 0.15
        else:
                max_eps = 0.30

	veri_directory = "CW"


import math
from statsmodels.stats.proportion import proportion_confint
#from IPython.display import clear_output, display

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
                       m_delta, out=None, max_k=1, attacking=False):
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
            start = 0
            model.layers[start].set_weights([m['arr_0'][0],m['arr_0'][1]])
            model.layers[start+2].set_weights([m['arr_0'][2], m['arr_0'][3]])
            model.layers[start+3].set_weights([m['arr_0'][4], m['arr_0'][5]])
        result = testproperty(inp, out, model, m_delta, max_k=max_k)
        if(result == -1):
            misses += 1
            result = 0
        successes += result
        iterations += 1
        # Setting the method equal to 'beta' here gives us clopper-pearson
        # and ensures that these bounds are exact.
        lb, ub = proportion_confint(successes, iterations, method='beta')
        if(math.isnan(lb)):
            lb = 0 # Clipping the lowerbound so if it is nan we set it to zero
        I = [lb, ub]
        hb = absolute_massart_halting(successes, iterations, I, epsilon)
        if(hb == -1):
            halting_bound = chernoff_bound
        else:
            halting_bound = min(hb, chernoff_bound)
    print("Exited becuase %s >= %s"%(iterations, halting_bound))
    return successes/iterations, misses/iterations


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer, Flatten
from keras.layers import Reshape
import time

model = Sequential()
model.add(Conv2D(25, (3, 3), activation='relu', input_shape=(28,28,channels)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(desired_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print model.summary()


from os import listdir
from os.path import isfile, join

# Load in the model weights from this training
weights_path = weights_path + '/'
model_weights = [weights_path + f for f in listdir(weights_path) if isfile(join(weights_path, f))]
print len(model_weights)



epsilon = 0.075
delta = 0.075
alpha = 0.05
x_test = X_test
m_delta = 0.75 #-1 #max_eps/10.0
sess = backend.get_session()

if not os.path.exists(model_path + "Results/"):
        os.mkdir(model_path + "Results/")

if not os.path.exists(model_path + "Results/%s_Attacks/"%(veri_directory)):
        os.mkdir(model_path + "Results/%s_Attacks/"%(veri_directory))




attack_locals = []
start = time.time()
val, misses = sequential_massart(epsilon, delta, alpha, model_weights, 
                             verifier, x_test[image_num], m_delta, max_k=0.25, attacking=True)
end = time.time()
# Need to save these
try:
     avg = sum(attack_locals)/len(attack_locals)
except:
    avg = np.zeros((28,28,3))
avg_b,avg_g,avg_r = cv2.split(np.reshape(avg, (28,28,3)))
variation = np.zeros((28,28,3))
for i in attack_locals:
    variation += (avg-i)**2
try:    
    variation/=N
except:
    print("Divide by zero... but continuing")

var_b,var_g,var_r = cv2.split(np.reshape(variation, (28,28,3)))
f=open(model_path + "Results/Stats-subfig-%s.txt"%(veri_directory), "a+")
f.write("| %s ; %s ; %s ; (%s) |"%(image_num,val,end-start,max_eps))
if not os.path.exists(model_path + "Results/%s_Attacks/image_%s_eps_%s"%(veri_directory, image_num,max_eps)):
    os.mkdir(model_path +"Results/%s_Attacks/image_%s_eps_%s"%(veri_directory, image_num,max_eps))
#np.savetxt(model_path + "FGSM_Attacks/image_%s_eps_%s/original_image_val=%s.txt"%(image_num, max_eps, val), x_test[image_num])
cv2.imwrite(model_path + "Results/%s_Attacks/image_%s_eps_%s/original_image_val=%s.png"%(veri_directory, image_num, max_eps, val), x_test[image_num]+0.5)

np.savetxt(model_path + "Results/%s_Attacks/image_%s_eps_%s/average_manip_b.txt"%(veri_directory, image_num,max_eps), avg_b)
np.savetxt(model_path + "Results/%s_Attacks/image_%s_eps_%s/average_manip_g.txt"%(veri_directory,image_num,max_eps), avg_g)
np.savetxt(model_path + "Results/%s_Attacks/image_%s_eps_%s/average_manip_r.txt"%(veri_directory,image_num,max_eps), avg_r)


np.savetxt(model_path + "Results/%s_Attacks/image_%s_eps_%s/variance_b.txt"%(veri_directory,image_num,max_eps), var_b)
np.savetxt(model_path + "Results/%s_Attacks/image_%s_eps_%s/variance_g.txt"%(veri_directory,image_num,max_eps), var_g)
np.savetxt(model_path + "Results/%s_Attacks/image_%s_eps_%s/variance_r.txt"%(veri_directory,image_num,max_eps), var_r)

