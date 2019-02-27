# Author: Matthew Wicker

import skimage
from skimage import transform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import copy
from copy import deepcopy
import cleverhans
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import CarliniWagnerL2
from keras import backend
import numpy as np
import cv2
sess = backend.get_session()
channels = 3
P_NORM = 2 #float("inf")
attack_locals = []

"""
This proceedure is slightly different as we are going to not fix delta
prior to the massart algorithm (we will ignore if there is one already 
set) and we will return the pixel in the image that causes the greatest
chage to the softmax layer wrt the predefined P_NORM value. This can
then be chained as a greedy search for adversarial examples that 
does partial exhaustion of the input space.

-- This is a future aim.
"""

def single_pix_verifier(inp, target, model, m_delta, max_k=1):
    target = np.squeeze(model.predict(np.reshape(inp, (1,28,28,channels))))
    #if(np.argmax(value) != np.argmax(target)):
    #    return -1
    for i in range(28*28):
        for j in range(28*28):
            for z in range(3):
                inp[i][j][z] = 1 - inp[i][j][z]
                image = inp[i][j][z]
                adv_x = image
                value = np.squeeze(model.predict(np.reshape(image, (1,28,28,channels))))

                if(m_delta == -1):
                    draw = np.random.uniform()
                    v,t= 0,0
                    if(value[0] < draw):
                        v = 1
                    if(target[0] < draw):
                        t = 1
                    if(t != v):
                        adv_x = np.squeeze(adv_x)
                        ex = np.reshape(inp,(28,28,3))
                        attack_locals.append(adv_x-ex)
                        return 0

                elif(np.linalg.norm(value - target, ord=P_NORM) >= m_delta):
                    adv_x = np.squeeze(adv_x)
                    ex = np.reshape(inp,(28,28,3))
                    attack_locals.append(adv_x-ex)
                    return 0
                inp[i][j][z] = 1 - inp[i][j][z]
    return 1

def translational_verifier(inp, target, model, m_delta, max_k=1):
    key = copy.deepcopy(inp)
    image = inp[max_k:len(inp)-max_k, max_k:len(inp[0])-max_k]
    image = cv2.resize(image,(28,28))
    target = np.squeeze(model.predict(np.reshape(image, (1,28,28,channels))))
    target = copy.deepcopy(target)
    for i in range(-1*max_k,max_k):
        for j in range(-1*max_k,max_k):
            tform = SimilarityTransform(translation=(i, j))
            warped = warp(key, tform)
            warped = warped[max_k:len(warped)-max_k,max_k:len(warped[0])-max_k]
            warped = cv2.resize(warped,(28,28))
            warped = np.asarray([warped])
            value = np.squeeze(model.predict(np.reshape(warped, (1,28,28,3))))
            
            if(m_delta == -1):
                draw = np.random.uniform()
                v,t= 0,0
                if(value[0] < draw):
                    v = 1
                if(target[0] < draw):
                    t = 1
                if(t != v):
                    adv_x = np.squeeze(warped)
                    ex = np.reshape(inp,(28,28,3))
                    attack_locals.append(adv_x-ex)
                    return 0
            elif(np.linalg.norm(value - target, ord=P_NORM) >= m_delta):
                adv_x = np.squeeze(warped)
                ex = np.reshape(inp,(28,28,3))
                attack_locals.append(adv_x-ex)
                return 0
    return 1

def rotational_verifier(inp, target, model, m_delta, max_k=1):
    crop = 4
    use = inp[crop:len(inp)-crop, crop:len(inp[0])-crop]
    use = cv2.resize(use,(28,28))
    target = np.squeeze(model.predict(np.reshape(inp, (1,28,28,channels))))
    for i in range(-1*max_k, max_k):
        image = np.reshape(inp, (28,28,3))#np.squeeze(inp)
        image = transform.rotate(image, i*5)
        image = image[crop:len(image)-crop, crop:len(image[0])-crop]
        image = cv2.resize(image,(28,28))
        image = np.asarray([image])
        adv_x = image
        value = np.squeeze(model.predict(np.reshape(image, (1,28,28,channels))))

        if(m_delta == -1):
            draw = np.random.uniform()
            v,t= 0,0
            if(value[0] < draw):
                v = 1
            if(target[0] < draw):
                t = 1
            if(t != v):
                adv_x = np.squeeze(adv_x)
                ex = np.reshape(inp,(28,28,3))
                attack_locals.append(adv_x-ex)
                return 0

        elif(np.linalg.norm(value - target, ord=P_NORM) >= m_delta):
            adv_x = np.squeeze(adv_x)
            ex = np.reshape(inp,(28,28,3))
            attack_locals.append(adv_x-ex)
            return 0
    return 1



def FGSM_verifier(inp, target, model, m_delta, max_k=1):
    target = np.squeeze(model.predict(np.reshape(inp, (1,28,28,channels))))
    fgsm_params = {'eps':max_k,'clip_min':-0.5,'clip_max':0.5}
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    adv_x = fgsm.generate_np(np.reshape(inp, (1,28,28,channels)), **fgsm_params)
    value = np.squeeze(model.predict(np.reshape(adv_x, (1,28,28,channels))))
    
    if(m_delta == -1):
        draw = np.random.uniform()
        v,t= 0,0
        if(value[0] < draw):
            v = 1
        if(target[0] < draw):
            t = 1
        if(t != v):
            adv_x = np.squeeze(adv_x)
            ex = np.reshape(inp,(28,28,3))
            attack_locals.append(adv_x-ex)
            return 0
        else:
            return 1
    
    if(np.linalg.norm(value - target, ord=P_NORM) >= m_delta):
        adv_x = np.squeeze(adv_x)
        ex = np.reshape(inp,(28,28,3))
        attack_locals.append(adv_x-ex)
        return 0
    else:
        return 1
    
    
def PGD_verifier(inp, target, model, m_delta, max_k=1):
    target = np.squeeze(model.predict(np.reshape(inp, (1,28,28,channels))))
    pgd_params = {'eps':max_k, 'eps_iter':max_k/2, 'clip_min':-0.5,'clip_max':0.5,'nb_iter':10}
    wrap = KerasModelWrapper(model)
    pgd = ProjectedGradientDescent(wrap, sess=sess)
    adv_x = pgd.generate_np(np.reshape(inp, (1,28,28,channels)), **pgd_params)
    value = np.squeeze(model.predict(np.reshape(adv_x, (1,28,28,channels))))
    
    if(m_delta == -1):
        draw = np.random.uniform()
        v,t= 0,0
        if(value[0] < draw):
            v = 1
        if(target[0] < draw):
            t = 1
        if(t != v):
            adv_x = np.squeeze(adv_x)
            ex = np.reshape(inp,(28,28,3))
            attack_locals.append(adv_x-ex)
            return 0
        else:
            return 1
    
    if(np.linalg.norm(value - target, ord=P_NORM) >= m_delta):
        adv_x = np.squeeze(adv_x)
        ex = np.reshape(inp,(28,28,3))
        attack_locals.append(adv_x-ex)
        return 0
    else:
        return 1
    
    
    
def CWL2_verifier(inp, target, model, m_delta, max_k=1):
    target = np.squeeze(model.predict(np.reshape(inp, (1,28,28,channels))))
    cw_params = {'clip_min':-0.5,'clip_max':0.5}
    wrap = KerasModelWrapper(model)
    cw = CarliniWagnerL2(wrap, sess=sess)
    adv_x = cw.generate_np(np.reshape(inp, (1,28,28,channels)), **cw_params)
    value = np.squeeze(model.predict(np.reshape(adv_x, (1,28,28,channels))))
    value = np.squeeze(model.predict(np.reshape(adv_x, (1,28,28,channels))))
    
    if(m_delta == -1):
        draw = np.random.uniform()
        v,t= 0,0
        if(value[0] < draw):
            v = 1
        if(target[0] < draw):
            t = 1
        if(t != v):
            adv_x = np.squeeze(adv_x)
            ex = np.reshape(inp,(28,28,3))
            attack_locals.append(adv_x-ex)
            return 0
        else:
            return 1
    
    if(np.linalg.norm(value - target, ord=P_NORM) >= m_delta):
        adv_x = np.squeeze(adv_x)
        ex = np.reshape(inp,(28,28,3))
        attack_locals.append(adv_x-ex)
        return 0
    else:
        return 1

