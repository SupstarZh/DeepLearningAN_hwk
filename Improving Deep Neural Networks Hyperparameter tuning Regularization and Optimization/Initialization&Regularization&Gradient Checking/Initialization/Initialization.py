import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
fig = plt.figure()

train_x, train_y, test_x, test_y = load_dataset() #x1->rcosx x2->rsinx  y->label

fig = plt.figure()
plt.scatter(test_x[0, :], test_x[1,:], c=np.squeeze(test_y), s=40, cmap=plt.cm.Spectral)

def initialize_parameters_zeros(layer_dims):
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def initialize_parameters_random(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

def initialize_parameters_he(layer_dims):
    parameters = {}
    L = len(layer_dims);
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2.0/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    return parameters

def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization = "he"):
    grads = {}
    costs = []
    m = X.shape[1]
    layer_dims = [X.shape[0], 10, 5, 1]
    
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layer_dims)
    
    for i in range(num_iterations):
        a3, cache = forward_propagation(X, parameters)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}:{}".format(i, cost))
            costs.append(cost)
    plt.figure()
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundrens)')
    plt.title("Learning rate=" + str(learning_rate))
    plt.show()
    
    return parameters
    
if __name__ == '__main__':
    parameters = model(train_x, train_y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he")
    print("On the train set:")
    #plt.figure()
    predictions_train = predict(train_x, train_y, parameters)
    print("On the test set:")
    #plt.figure()
    predictions_test = predict(test_x, test_y, parameters)
    plt.close()
    
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x:predict_dec(parameters, x.T), train_x, train_y)
    
        
    