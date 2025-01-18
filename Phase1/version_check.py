# Desc - A simple program which checks the version of the 
# PyTorch library installed and checks whether 'CUDA' is available.

import torch

print("PyTorch version:", torch.__version__)

print("CUDA available:", torch.cuda.is_available())