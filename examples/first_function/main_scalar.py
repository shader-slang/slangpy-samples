# SPDX-License-Identifier: Apache-2.0

# First function example
# https://slangpy.shader-slang.org/en/latest/src/basics/firstfunctions.html

import slangpy as spy
import pathlib

# Create a SlangPy device that looks in the local folder for any Slang includes
device = spy.create_device(
    include_paths=[
        pathlib.Path(__file__).parent.absolute(),
    ]
)

# Load the module
module = spy.Module.load_from_file(device, "example.slang")

# Call the function and print the result
result = module.add(1.0, 2.0)
print(result)

# SlangPy also supports named parameters
result = module.add(a=1.0, b=2.0)
print(result)
