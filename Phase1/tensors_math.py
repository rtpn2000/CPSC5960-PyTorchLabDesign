# Desc - Performing various basic arithmetic operation on tensors.

import torch
import numpy as np

tensor1 = torch.tensor([1,2,3,4,5])
tensor2 = torch.tensor([6,7,8,9,10])

# Addition 
tensor_add = tensor1 + tensor2
print('Addition:', tensor_add)

# Division - Longhand else can be simply written as 'tensor2 / tensor1'.
tensor_div = torch.div(tensor2, tensor1)
print('Division:', tensor_div)

# Exponentiation
tensor_expo1 = tensor1 ** 2
print('Exponentiation: ', tensor_expo1)
# Exponentiation - Longhand
tensor_expo_2 = torch.pow(tensor2, 2)
print('Exponentiation - Longhand:', tensor_expo_2)

# Matrix Multiplication
tensor_a = torch.tensor(([1,2], [3,4]))
tensor_b = torch.tensor(([9,8], [7,6]))
res = torch.matmul(tensor_a, tensor_b)
print('\nMatrix Multiplication:', res)