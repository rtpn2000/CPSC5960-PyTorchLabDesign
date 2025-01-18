# Desc - Exploring tensors!

import torch
import numpy as np

# Initializing a simple tensor.
tensor1 = torch.tensor([1,2])
print('\nA basic tensor:\n', tensor1)

# Initializing a 2D tensor.
tensor2 = torch.tensor([[1,2], [3,4], [5,6]])
print('\nA basic 2D tensor\n:', tensor2)

# Initializing a tensor from a list - 'data'.
data = [[1,2], [3,4]]
tensor3 = torch.tensor(data)
print('\nA tensor initialized from a list:\n', tensor3)

# Initializing a tensor from a numpy array.
npArray_data = np.array([1,2,3,4])
tensor4 = torch.tensor(npArray_data)
print('\nA tensor initialized from a numpy array:\n', tensor4)