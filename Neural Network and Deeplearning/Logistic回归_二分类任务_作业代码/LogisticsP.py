import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import practice_part1 as pybasic

#The main steps for building a Neural Network are: 
#1. Define the model structure (such as number of input features) 
#2. Initialize the model’s parameters 
#3. Loop: 
#- Calculate current loss (forward propagation) 
#- Calculate current gradient (backward propagation) 
#- Update parameters (gradient descent)
#
#You often build 1-3 separately and integrate them into one function we call model().


# w初始化为全零列向量
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

# 正向传播和反向传播 sigmod和cost,dw,db
def propagate(w, b, X, Y):
    m = X.shape[1]
    z = np.dot(w.T, X)+b
    A = pybasic.basic_sigmoid(z)
    cost = -(1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    dw = (1.0/m)*np.dot(X, (A-Y).T)
    db = (1.0/m)*np.sum(A-Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)

    grads = {"dw": dw, "db":db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
      Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        if i%100 == 0:
            costs.append(cost)
        
        if print_cost and i%100 == 0:
            print("Cost after iteration %i:%f" %(i, cost))
    params = {"w":w,"b":b}
    grads = {"dw":dw, "db":db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = pybasic.basic_sigmoid(np.dot(w.T, X)+b)
    
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

#What to remember: 
#You’ve implemented several functions that: 
#- Initialize (w,b) 
#- Optimize the loss iteratively to learn parameters (w,b): 
#- computing the cost and its gradient 
#- updating the parameters using gradient descent 
#- Use the learned (w,b) to predict the labels for a given set of examples

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    
    prarms, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = prarms["w"]
    b = prarms["b"]
    
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    
    print("train accuracy:{} %".format(100- np.mean(np.abs(Y_prediction_train - Y_train))*100))
    print("test accuracy:{} %".format(100- np.mean(np.abs(Y_prediction_test - Y_test))*100))
    
    d = {"costs":costs,
         "Y_prediction_train": Y_prediction_train,
         "Y_prediction_test": Y_prediction_test,
         "w":w ,
         "b":b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost = True)

index = 33
plt.imshow(test_set_x[:, index].reshape((64,64,3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")
plt.close()

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
plt.close()


#learning_rates = [0.01, 0.001, 0.0001]
#models = {}
#for i in learning_rates:
#    print("learning rate is:" + str(i))
#    models[str(i)] = model(train_set_x, train_set_y, test_set_x,  test_set_y, num_iterations=1500, learning_rate = i, print_cost = False)
#    print('\n'+"------------------------------------"+'\n')
#    
#for i in learning_rates:
#    plt.plot(np.squeeze(models[str(i)]["costs"]), label = str(models[str(i)]["learning_rate"]))
#    
#plt.ylabel('cost')
#plt.xlabel('iterations')
#
#legend = plt.legend(loc='upper_center', shadow=True)
#frame = legend.get_frame()
#frame.set_facecolor('0.90')
#plt.show()