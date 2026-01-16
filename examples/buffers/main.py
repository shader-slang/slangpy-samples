# SPDX-License-Identifier: Apache-2.0

# Buffers example
# https://slangpy.shader-slang.org/en/latest/src/basics/tensors.html
# This example requires tev (https://github.com/Tom94/tev) to display results.

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

# Create a couple of 2D 16x16 tensors
image_1 = spy.Tensor.empty(device,shape=(16, 16), dtype=module.Pixel)
image_2 = spy.Tensor.empty(device,shape=(16, 16), dtype=module.Pixel)

# Use a cursor to fill the first tensor with readable structured data.
cursor_1 = image_1.cursor()
for x in range(16):
    for y in range(16):
        cursor_1[x + y * 16].write(
            {
                "r": (x + y) / 32.0,
                "g": 0,
                "b": 0,
            }
        )
cursor_1.apply()

# Use the fact that we know the tensors are just 16x16 grids of 3 floats
# to just populate the 2nd tensor straight from random numpy array
image_2.copy_from_numpy(0.1 * np.random.rand(16 * 16 * 3).astype(np.float32))

# Call the module's add function
result = module.add(image_1, image_2)

# Use a cursor to read and print pixels (would also be readable in the watch window)
result_cursor = result.cursor()
for x in range(16):
    for y in range(16):
        pixel = result_cursor[x + y * 16].read()
        print(f"Pixel ({x},{y}): {pixel}")

# Or if installed, we can use tev to show the result (https://github.com/Tom94/tev)
bitmap = spy.Bitmap(data=result.to_numpy().view(np.float32))
spy.tev.show(bitmap)
