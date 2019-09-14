import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters, grads, learning_rate):
    lens = len(parameters) // 2
    for l in range(lens):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]
        
    return parameters

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    
    #同步打乱X、Y顺序 Shuffling
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    
    #Partitioning 按batch-size同步划分X、Y
    num_compelete_minibatches = math.floor(m/mini_batch_size)
    for k in range(num_compelete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_compelete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_compelete_minibatches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
#- Shuffling and Partitioning are the two steps required to build mini-batches
#- Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.
def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(L):
        v["dW"+str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        v["db"+str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
    
    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        v["dW"+str(l+1)] = beta*v["dW"+str(l+1)] + (1-beta)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta*v["db"+str(l+1)] + (1-beta)*grads["db"+str(l+1)]
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*v["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*v["db"+str(l+1)]
        
    return parameters,v
#- Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
#- You have to tune a momentum hyperparameter ββ and a learning rate αα.
def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        v["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
        s["dW" + str(l+1)] = np.zeros((parameters["W"+str(l+1)].shape))
        s["db" + str(l+1)] = np.zeros((parameters["b"+str(l+1)].shape))
    
    return v, s

#一次adam 参数更新
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_correct = {}
    s_correct = {}
    for l in range(L):
        v["dW"+str(l+1)] = beta1*v["dW"+str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]
        
        s["dW"+str(l+1)] = beta2*s["dW"+str(l+1)] + (1-beta2)*np.square(grads["dW"+str(l+1)])
        s["db"+str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2)*np.square(grads["db"+str(l+1)])
        
        v_correct["dW"+str(l+1)] = v["dW"+str(l+1)]/(1-beta1**t)
        v_correct["db"+str(l+1)] = v["db"+str(l+1)]/(1-beta1**t)
        s_correct["dW"+str(l+1)] = s["dW"+str(l+1)]/(1-beta2**t)
        s_correct["db"+str(l+1)] = s["db"+str(l+1)]/(1-beta2**t)
        
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*v_correct["dW"+str(l+1)]/np.sqrt(s_correct["dW"+str(l+1)]+epsilon)
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*v_correct["db"+str(l+1)]/np.sqrt(s_correct["db"+str(l+1)]+epsilon)
    
    return parameters, v, s

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10
    parameters = initialize_parameters(layers_dims)
    
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    for i in range(num_epochs):
        seed = seed+1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        
        for minibatch in mini_batches:
            (minibatch_X, minibatch_Y) = minibatch

            a3, caches = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(a3, minibatch_Y)
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t+1
                parameters, v, s = update_parameters_with_adam(parameters,grads, v, s, t, learning_rate,
                                                                   beta1, beta2, epsilon)
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i:%f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("epochs(per 100)")
    plt.title("Learning_rate = " + str(learning_rate))
    plt.show()
    
    return parameters

if __name__ == "__main__":
    train_X, train_Y = load_dataset()
    plt.close()
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")
    
    predictions = predict(train_X, train_Y, parameters)
    
    plt.title("Model with Gradient optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    plot_decision_boundary(lambda x:predict_dec(parameters, x.T), train_X, train_Y)
    
    
    parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")
    # Predict
    predictions = predict(train_X, train_Y, parameters)
    # Plot decision boundary
    plt.title("Model with Momentum optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

    parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")
    # Predict
    predictions = predict(train_X, train_Y, parameters)
    # Plot decision boundary
    plt.title("Model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
#- Usually works well even with little tuning of hyperparameters (except αα)