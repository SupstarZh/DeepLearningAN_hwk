import math
import numpy as np
import time

def basic_sigmoid(x):
     s= 1.0 / (1+np.exp(-x))
     return s
 
def sigmoid_derivative(x):
    s = 1.0/(1+1/np.exp(x))
    ds = s*(1-s)
    return ds

def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    return v

def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_normalized = x/x_norm
    return x_normalized

def softmax(x):
    x_exp = np.exp(x)
    row_sum = np.sum(x_exp, axis=0, keepdims = True)
    x_softmax = x_exp/row_sum
    return x_softmax

#- np.exp(x) works for any np.array x and applies the exponential function to every coordinate 
#- the sigmoid function and its gradient 
#- image2vector is commonly used in deep learning 
#- np.reshape is widely used. In the future, you’ll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs. 
#- numpy has efficient built-in functions 
#- broadcasting is extremely useful
    
def L1(yhat, y):
    loss = np.sum(np.abs(yhat-y))
    return loss

def L2(yhat, y):
    loss = np.sum(np.dot(yhat-y, yhat-y)) # np.dot -> np.power(x,2) is also right
    return loss

#- Vectorization is very important in deep learning. It provides computational efficiency and clarity. 
#- You have reviewed the L1 and L2 loss. 
#- You are familiar with many numpy functions such as np.sum, np.dot, np.multiply, np.maximum, etc…

# General Architecture of the learning algorithm
    
#Key steps: 
#In this exercise, you will carry out the following steps: 
#- Initialize the parameters of the model 
#- Learn the parameters for the model by minimizing the cost 
#- Use the learned parameters to make predictions (on the test set) 
#- Analyse the results and conclude