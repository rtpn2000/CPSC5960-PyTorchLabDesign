# Desc - Displaying some of the basic attributes of tensors.

import torch

# Initializing a basic tensor!
tensor = torch.tensor([[1,2,3], [6,5,4], [20,30,40]])

print("Shape of tensor:", tensor.shape)
print("Size of tensor:", tensor.size())
print("Data type of tensor:", tensor.dtype)
print("Device of tensor:", tensor.device)
print("Number of elements in tensor:", tensor.numel())
print("Number of dimensions in tensor:", tensor.ndimension())