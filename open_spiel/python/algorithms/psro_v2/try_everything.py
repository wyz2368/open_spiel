import numpy as np
import copy



def softmax(x, temperature=1/5):
    return np.exp(x / temperature)/np.sum(np.exp(x / temperature))


x = np.array([0.3,0,0])
print(softmax(x))