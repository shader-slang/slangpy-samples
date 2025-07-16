# SPDX-License-Identifier: Apache-2.0

# First function example
# https://slangpy.shader-slang.org/en/latest/src/basics/returntype.html

import slangpy as spy
import pathlib
import numpy as np

# Create a device with the local folder for slangpy includes
device = spy.create_device(
    include_paths=[
        pathlib.Path(__file__).parent.absolute(),
    ]
)

# Load module
module = spy.Module.load_from_file(device, "example.slang")

# Create a couple of buffers with 128x128 random floats
a = np.random.rand(128, 128).astype(np.float32)
b = np.random.rand(128, 128).astype(np.float32)

# Call our function and ask for a texture back
result = module.add(a, b, _result='texture')

# Print the first 5x5 values
print(result.to_numpy()[:5, :5])

# Display the result using tev
spy.tev.show(result, name='add random')
