import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import model_from_json
import numpy as np
from tqdm import trange
from os import listdir
from os.path import isfile, join

def sample_networks(path_to_model, path_to_model_weights, num_models=100, tmp_path='SampledModels'):
    # First we need to check how many models have been completed up to this point...
    # Then use that number to continue to append
    try:
    	current_models = [f for f in listdir(tmp_path) if isfile(join(tmp_path, f))]
   	model_numbers = [int(s.split('_')[-1][:-3]) for s in current_models]
    	_comp = max(model_numbers)
    except:
	print "Starting sampling for the first time..."
    	_comp = 0
    for model_number in trange(_comp, _comp + num_models):
        json_file = open('mnist-mlp.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("mnist-mlp.h5")
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        layer_num, real_layer = 0, 0
        dropout_layers, dropout_rates, real_layers = [], [], []
        for layer in model.layers:
            w = np.asarray(layer.get_weights())
            if(w.shape[-1] >= 2):
                real_layer = layer_num
            if('dropout' in layer.name):
                real_layers.append(real_layer)
                dropout_layers.append(layer_num)
                dropout_rates.append(layer.rate)
            layer_num+=1
            
        for i in range(len(dropout_layers)):
            weights = np.asarray(model.layers[real_layers[i]].get_weights()[0])
            biases  = np.asarray(model.layers[real_layers[i]].get_weights()[1])
            shape_of_neuron_input = weights.shape[:-1]
            #print("We have a layer with %s many neurons each with input shape %s for this layer we want to mask %s with zeros"
            #      %(biases.shape[0], shape_of_neuron_input, dropout_rates[i]))

            # Now we need to create a mask. 
            remove = np.random.permutation(biases.shape[0])
            remove_to = int(dropout_rates[i] * biases.shape[0])

            keep = np.ones(shape_of_neuron_input) + dropout_rates[i] 
            delete = np.zeros(shape_of_neuron_input)

            weights = np.moveaxis(weights,-1,0)
            counter = 0
            for j in remove:
                if(counter <= remove_to):
                    weights[j] *= delete
                    biases[j] *= 0
                else:
                    weights[j] *= keep
                    biases[j] *= 1+dropout_rates[i]
                counter += 1
            weights = np.moveaxis(weights,0,-1)
            model.layers[real_layers[i]].set_weights([weights,biases])
        # Save the model that we sampled
        model.save_weights(tmp_path + "/sampled_model_%s.h5"%(model_number))
        #gc.collect()
sample_networks('mnist-mlp.json', 'mnist-mlp.h5', num_models=50)
