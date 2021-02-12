import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_d(x):
    return (x * (1 - x))

def relu(x):
    result = np.maximum(0, x)
    return result

def relu_d(x):
    result = x
    result[result <= 0] = 0
    result[result > 0] = 1
    return result

