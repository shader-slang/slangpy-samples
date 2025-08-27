# SPDX-License-Identifier: Apache-2.0

# PyTorch example
# https://slangpy.shader-slang.org/en/latest/src/autodiff/pytorch.html

import slangpy as spy
import pathlib
import torch

if not torch.cuda.is_available():
    print("CUDA is not available, skipping torch example")
    exit(0)

# Create a device configured for PyTorch integration
# CUDA backend is recommended for best performance
device = spy.create_torch_device(
    type=spy.DeviceType.cuda,
    include_paths=[
        pathlib.Path(__file__).parent.absolute(),
    ]
)

# Load module using the standard Module type
# SlangPy automatically detects PyTorch tensors and enables auto-grad support
module = spy.Module.load_from_file(device, "example.slang")

# Create a tensor
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device="cuda", requires_grad=True)

# Evaluate the polynomial. Result will automatically be a torch tensor.
# Expecting result = 2x^2 + 8x - 1
result = module.polynomial(a=2, b=8, c=-1, x=x)
print(result)

# Run backward pass on result, using result grad == 1
# to get the gradient with respect to x
result.backward(torch.ones_like(result))
print(x.grad)
