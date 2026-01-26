from mytorch import Tensor
import mytorch.nn as nn

from mytorch.nn import functional as F
from mytorch.nn import Sigmoid, ReLU, Softmax, Tanh, GELU

x = Tensor([-1000, 1000])
result = Tanh()(x)

x = Tensor([-1.0])
result = Tanh()(x)


x = Tensor([[1,2,3]])
result = Softmax()(x)
print(result)




