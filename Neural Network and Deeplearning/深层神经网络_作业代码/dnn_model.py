import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
#def initialize_parameters_deep(layer_dims):
#    ...
#    return parameters 
#def L_model_forward(X, parameters):
#    ...
#    return AL, caches
#def compute_cost(AL, Y):
#    ...
#    return cost
#def L_model_backward(AL, Y, caches):
#    ...
#    return grads
#def update_parameters(parameters, grads, learning_rate):
#    ...
#    return parameters
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

#train_x.shape = (12288, 209) 12288=64*64*3
train_x = train_x_flatten/255
test_x = test_x_flatten/255

layers_dims = [12288, 20, 7, 5, 1] #5-layer model

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []    
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i%100 == 0:
            print("Cost after iteration {}:{}" .format(i, np.squeeze(cost)))
            costs.append(cost)
    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per tennns)')
    plt.title("Learning rate=" + str(learning_rate))
    plt.show()
    return parameters

if __name__ == '__main__':
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)
    print_mislabeled_images(classes, test_x, test_y, pred_test)