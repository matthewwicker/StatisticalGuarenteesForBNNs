# Author: Matthew Wicker
import numpy as np
import cv2
import pickle
from keras.utils import np_utils

def load(training_file, validation_file, testing_file):
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

    def normalize_image(image):
        return -0.5 + (image*1.0)/(255)

    def Gray_image(image):
        if(channels == 1):
            return np.resize(cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)[:,:,0],(28,28,1))
        return image

    def preprocess(image):
        img= []
        for i in image:
            img.append(normalize_image(Gray_image(i)))
            #mg.append(Gray_image)
        img = np.array(img)
        return img

    X_train = preprocess(X_train)
    X_valid = preprocess(X_valid)
    X_test  = preprocess(X_test)

    # 6. Preprocess class labels
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    y_valid = np_utils.to_categorical(y_valid, n_classes)

    numeric = np.argmax(y_train,axis=1)
    counts = np.bincount(numeric)
    classes_to_use = counts.argsort()[-10:][::-1]
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
    return X_train, y_train, X_test, y_test
