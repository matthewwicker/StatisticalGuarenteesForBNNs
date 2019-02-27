#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:48:09 2019

@author: apatane
"""
import numpy as np

def averagePooling(dataMatrix):

    pooledData = np.zeros((len(dataMatrix),14,14),dtype = dataMatrix[0].dtype);

    for h in range(len(dataMatrix)): 
        x = dataMatrix[h]
        y = np.zeros((14,14),dtype = x.dtype)
        for i in range(14):
            i_4_x = [i*2,i*2+1]
            for j in range(14):
                j_4_x = [j*2,j*2+1]
                #print x.shape
                #print i_4_x
                #print j_4_x
                aux = 0
                for i1 in range(len(i_4_x)):
                    for j1 in range(len(j_4_x)):
                        aux = aux + x[i_4_x[i1],j_4_x[j1]]
                y[i,j] =  aux/4.0
        pooledData[h] = y
    return pooledData



def keep_only_spec_labels(x,y,label0,label1):
    
    x_0 = x[y == label0]
    x_1 = x[y == label1]
    x = np.concatenate( (x_0,x_1)  )
    
    y_0 = y[y == label0]
    y_1 = y[y== label1]
    y = np.concatenate( (y_0,y_1)  )
    return x,y

