# SPDX-License-Identifier: Apache-2.0

import slangpy as spy
import numpy as np
import time
import math
from pathlib import Path
import sys
from app import App

sys.path.append(str(Path(__file__).parent.parent))

from neuralnetwork import neuralnetworks as nn

# Python implementation of the neural material
class NeuralMaterial(nn.IModel):
    def __init__(self, num_material_params: int, num_latents: int, use_coopvec: bool):
        super().__init__()

        self.num_material_params = num_material_params
        self.num_latents = num_latents

        # If the device supports cooperative vector, run MLP at half precision and in coopvec mode
        if use_coopvec:
            mlp_input = nn.ArrayKind.coopvec
            mlp_precision = nn.Real.half
        else:
            mlp_input = nn.ArrayKind.array
            mlp_precision = nn.Real.float

        # First, we instantiate the encoder network
        # This takes the material parameters and transforms them to a latent code
        self.encoder = nn.ModelChain(
            nn.Convert.to_precision(mlp_precision),
            nn.Convert.to_array_kind(mlp_input),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=64),
            nn.LeakyReLU(),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=64),
            nn.LeakyReLU(),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=64),
            nn.LeakyReLU(),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=num_latents),
            nn.LeakyReLU(),
            nn.Convert.to_float(),
            nn.Convert.to_array()
        );

        # Then, we instantiate the decoder network
        # This takes the latent code and view/light directions and returns the RGB material response
        self.decoder = nn.ModelChain(
            nn.Convert.to_precision(mlp_precision),
            nn.Convert.to_array_kind(mlp_input),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=32),
            nn.LeakyReLU(),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=32),
            nn.LeakyReLU(),
            nn.LinearLayer(num_inputs=nn.Auto, num_outputs=3),
            nn.Convert.to_vector(),
            nn.Convert.to_float(),
            nn.Exp()
        )

    def model_init(self, module: spy.Module, input_type):
        # Recursively initialize the networks
        self.encoder.initialize(module, f'float[{self.num_material_params}]')
        self.decoder.initialize(module, f'float[{6 + self.num_latents}]')

    @property
    def type_name(self) -> str:
        # We get the slang type for this material by plugging in the types
        # of the encoder/decoder networks
        return f"NeuralMaterial<{self.encoder.type_name}, {self.decoder.type_name}>"

    def get_this(self):
        # When passing this python class to slang, just pass the encoder/decoder networks to the slang struct
        return {
            "encoder": self.encoder.get_this(),
            "decoder": self.decoder.get_this(),
            "_type": self.type_name
        }

    # Book keeping for the neural networks library
    def children(self):
        return [self.encoder, self.decoder]

    def resolve_input_type(self, module: spy.Module):
        return module.MaterialTrainInput

def main():
    device = spy.create_device(spy.DeviceType.vulkan, False, include_paths=nn.slang_include_paths())

    app = App(device, "Neural Materials", 1024, 1024)

    # Load the textures of the reference material we want to learn
    loader = spy.TextureLoader(device)
    material_textures = {
        "albedoMap": loader.load_texture("PavingStones070_2K.diffuse.jpg", {"load_as_srgb": True}),
        "metalRoughMap": loader.load_texture("PavingStones070_2K.roughness.jpg", {"load_as_srgb": False}),
        "normalMap": loader.load_texture("PavingStones070_2K.normal.jpg", {"load_as_srgb": False}),
        "sampler": device.create_sampler(spy.TextureFilteringMode.linear, spy.TextureFilteringMode.linear),
        "_type": "TexturedMaterial"
    }

    # Check for coop-vec support. This will make the sample run much faster.
    if spy.Feature.cooperative_vector in device.features:
        print("Cooperative vector enabled!")
        use_coopvec = True
    else:
        print("Device does not support cooperative vector. Sample will run, but it will be slow")
        use_coopvec = False

    # Load the slang file containing our training code
    module = spy.Module.load_from_file(device, "NeuralMaterial.slang")

    # Instantiate and initialize the neural material
    neural_material = NeuralMaterial(num_material_params=9, num_latents=8, use_coopvec=use_coopvec)
    neural_material.initialize(module)

    # Set up the training hyperparameters
    learning_rate = 0.0005
    batch_shape = (256, 256)
    num_batches_per_epoch = 10

    # Set up the optimizer
    optimizer = nn.AdamOptimizer(learning_rate=learning_rate)
    # If we use coop-vec, the model runs at half precision. To avoid numerical issues,
    # we scale up the gradients by a factor of 128, and wrap the optimizer in a
    # FullPrecisionOptimizer, which avoids re-rounding to half at each iteration
    if use_coopvec:
        grad_scale = 512.0
        optimizer = nn.FullPrecisionOptimizer(optimizer, grad_scale)
    else:
        grad_scale = 1.0
    optimizer.initialize(module, neural_material.parameters())
    # Scale up the loss scale to compensate for the gradient scaling
    loss_scale = grad_scale / math.prod(batch_shape)

    # Create an array of random number generators, one for each entry in the batch
    seeds = np.random.get_bit_generator().random_raw(batch_shape).astype(np.uint32)
    rng = module.RNG(seeds)

    iteration = 0
    while app.process_events():
        start = time.time()

        # Submit a batch of training iterations to a command encoder and submit all at once for performance
        cmd = device.create_command_encoder()
        for i in range(num_batches_per_epoch):
            # Train material and compute gradients
            module.trainMaterial(material_textures, neural_material, rng, loss_scale, _append_to=cmd)
            # Run the optimizer to update the parameters with the gradients
            optimizer.step(cmd)

        # Submit the command buffer and wait for it to finish for a good interactive experience.
        # This slows things down a lot though - headless training will run faster, at the cost of interactivity
        id = device.submit_command_buffer(cmd.finish())
        device.wait_for_submit(id)

        # Render the current state of the neural material
        module.renderMaterial(material_textures, neural_material, iteration, pixel=spy.call_id(), _result=app.output)
        iteration += 1

        # Print stats about the current throughput
        elapsed = time.time() - start
        msamples = (num_batches_per_epoch * math.prod(batch_shape)) * 1e-6
        print(f"Iteration {iteration:5d} - "
              f"Throughput: {msamples / elapsed:.2f} MSamples/s "
              f"Epoch time: {elapsed * 1e3:.1f}ms")

        app.present()

if __name__ == "__main__":
    main()

