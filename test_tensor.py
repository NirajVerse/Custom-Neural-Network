from mytorch import Tensor
import mytorch.nn as nn

from mytorch.nn import functional as F
from mytorch.nn import Sigmoid, ReLU

# x = Tensor([-1,-2,-3,-4])
# y = Tensor([-1000, 1000])
# #result = x + y

# act = ReLU()(x)
# print(act)
# res = Sigmoid()(act)

# print(res)


import numpy as np


def test_unit_relu():
    """🔬 Test ReLU implementation."""
    print("🔬 Unit Test: ReLU...")

    relu = ReLU()

    # Test mixed positive/negative values
    x = Tensor([-2, -1, 0, 1, 2])
    result = relu.forward(x)
    expected = [0, 0, 0, 1, 2]
    assert np.allclose(result._data, expected), f"ReLU failed, expected {expected}, got {result._data}"

    # Test all negative
    x = Tensor([-5, -3, -1])
    result = relu.forward(x)
    assert np.allclose(result._data, [0, 0, 0]), "ReLU should zero all negative values"

    # Test all positive
    x = Tensor([1, 3, 5])
    result = relu.forward(x)
    assert np.allclose(result._data, [1, 3, 5]), "ReLU should preserve all positive values"

    # Test sparsity property
    x = Tensor([-1, -2, -3, 1])
    result = relu.forward(x)
    zeros = np.sum(result._data == 0)
    assert zeros == 3, f"ReLU should create sparsity, got {zeros} zeros out of 4"

    print("✅ ReLU works correctly!")

if __name__ == "__main__":
    test_unit_relu()