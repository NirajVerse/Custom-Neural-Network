import numpy as np 
from mytorch import Tensor

def sigmoid(x: Tensor):

    z = np.clip(x._data, -500, 500)
    result = np.zeros_like(z)

    pos_mark = z >=0

    result[pos_mark] = 1 / (1 + np.exp(-z[pos_mark]))


    #bug: when every thing is positive, this get exploded: need to fix this: bug fixed was using pos_mark instead of neg_mark

    neg_mark = z < 0
    result[neg_mark] = np.exp(z[neg_mark]) / (1 + np.exp(z[neg_mark]))  

    return Tensor(result)



def relu(x: Tensor):  ## implementing relu activation function: 
    return Tensor(np.maximum(0, x._data))   # as Relu just changes all the negatives into zeros and and all pos as it is