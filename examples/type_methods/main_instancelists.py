# SPDX-License-Identifier: Apache-2.0

# Type methods example
# https://slangpy.shader-slang.org/en/latest/src/basics/typemethods.html

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

# Create buffer of particles (.as_struct is used to make python typing happy!)
particles = spy.InstanceList(
    struct=module.Particle.as_struct(),
    data={
        "position": spy.NDBuffer(device, dtype=module.float3, shape=(10,)),
        "velocity": spy.NDBuffer(device, dtype=module.float3, shape=(10,)),
    },
)

# Construct every particle with position of 0, and use slangpy's rand_float
# functionality to supply a different rand vector for each one.
particles.construct(p=spy.float3(0), v=spy.rand_float(-1, 1, 3))

# Print all the particles by breaking them down into groups of 6 floats
print(particles.position.to_numpy().view(dtype=np.float32).reshape(-1, 3))
print(particles.velocity.to_numpy().view(dtype=np.float32).reshape(-1, 3))

# Update the particles
particles.update(0.1)

# Print all the particles by breaking them down into groups of 6 floats
print(particles.position.to_numpy().view(dtype=np.float32).reshape(-1, 3))
print(particles.velocity.to_numpy().view(dtype=np.float32).reshape(-1, 3))
