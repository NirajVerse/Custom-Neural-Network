from .module import Module
from .import functional as F
from mytorch import Tensor


## implementing this as a wrapper

class Sigmoid(Module):

    def forward(self, x:Tensor) -> Tensor:
        return F.sigmoid(x)
    

class ReLU(Module):
    
    def forward(self, x:Tensor) -> Tensor:
        return F.relu(x)


class Softmax(Module):

    def forward(self, x:Tensor) -> Tensor:
        return F.softmax(x)


class Tanh(Module):
    def forward(self, x:Tensor) -> Tensor:
        return F.tanh(x)

class GELU(Module):
    def forward(self, x:Tensor) -> Tensor:
        return F.gelu(x)
    
