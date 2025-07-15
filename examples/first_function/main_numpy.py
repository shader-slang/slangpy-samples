# SPDX-License-Identifier: Apache-2.0

# First function example
# https://slangpy.shader-slang.org/en/latest/src/basics/firstfunctions.html

import slangpy as spy
import pathlib
import numpy as np

# Create a SlangPy device; it will look in the local folder for any Slang includes
device = spy.create_device(
    include_paths=[
        pathlib.Path(__file__).parent.absolute(),
    ]
)

# Load the module
module = spy.Module.load_from_file(device, "example.slang")

# Create a couple of buffers containing 1,000,000 random floats
a = np.random.rand(1000000).astype(np.float32)
b = np.random.rand(1000000).astype(np.float32)

# Call our function and request a numpy array as the result (the default would be a buffer)
result = module.add(a, b, _result="numpy")

# Print the first 10 results
print(result[:10])
