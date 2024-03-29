import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

def layer_sizes(X, Y):
    n_x = X.shape[0] # 2
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y
#X_assess, Y_assess = layer_sizes_test_case()
#(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
#n_x, n_h, n_y = initialize_parameters_test_case()
#parameters = initialize_parameters(n_x, n_h, n_y)
    
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)*0.01
    W2 = np.random.randn(n_y, n_h)*0.01
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache
#X_assess, parameters = forward_propagation_test_case()
#A2, cache = forward_propagation(X_assess, parameters)
##print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    
    logprobs = np.multiply(np.log(A2), Y)+np.multiply(np.log(1-A2), 1-Y)
    cost = -(1.0/m)*np.sum(logprobs)
    
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    
    return cost
#A2, Y_assess, parameters = compute_cost_test_case()

#print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ_2 = A2 - Y
    dW_2 = (1.0/m)*np.dot(dZ_2, A1.T)
    db_2 = (1.0/m)*np.sum(dZ_2, axis=1, keepdims=True)
    dZ_1 = np.dot(W2.T, dZ_2) * (1-np.power(A1, 2))
    dW_1 = (1.0/m)*np.dot(dZ_1, X.T)
    db_1 = (1.0/m)*np.sum(dZ_1, axis=1, keepdims=True)
    grads = {"dW1": dW_1,
             "db1": db_1,
             "dW2": dW_2,
             "db2": db_2}

    return grads
#parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

#grads = backward_propagation(parameters, cache, X_assess, Y_assess)
#print ("dW1 = "+ str(grads["dW1"]))
#print ("db1 = "+ str(grads["db1"]))
#print ("dW2 = "+ str(grads["dW2"]))
#print ("db2 = "+ str(grads["db2"]))
    
def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    W2 = W2 - learning_rate*dW2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

#parameters, grads = update_parameters_test_case()
#parameters = update_parameters(parameters, grads)
#
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))

def nn_model(X, Y, n_h, num_iterations=10000, print_cost = False):
    np.random.seed(3)
    n_x, nh, n_y = layer_sizes(X, Y)
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        
        if print_cost and i%1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            
    return parameters
#X_assess, Y_assess = nn_model_test_case()
#parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2>0.5)
    return predictions

#parameters, X_assess = predict_test_case()

#predictions = predict(parameters, X_assess)
#print("predictions mean = " + str(np.mean(predictions)))
if __name__ == '__main__':
    #X, Y = load_planar_dataset()
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    dataset = "noisy_circles"
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    if dataset == "blobs":
        Y = Y%2
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral); 
    
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
    plt.title("Decision Boundary for hidden layer size " + str(4))
    
    predictions = predict(parameters, X)
    print('Accuracy:%d' % float((np.dot(Y, predictions.T)+np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')
    
