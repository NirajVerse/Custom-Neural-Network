from mytorch import Tensor
import mytorch.nn as nn

from mytorch.nn import functional as F
from mytorch.nn import Sigmoid, ReLU

x = Tensor([-1,-2,-3,-4])
y = Tensor([-1000, 1000])
#result = x + y

act = ReLU()(x)
print(act)
res = Sigmoid()(act)

print(res)

