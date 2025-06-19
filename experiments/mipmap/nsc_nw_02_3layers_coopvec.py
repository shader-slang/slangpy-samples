# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from app import App
import slangpy as spy
import numpy as np
import json

# Create the app and load the slang module.
app = App(width=512*3+10*2, height=512, title="Mipmap Example", device_type=spy.DeviceType.vulkan)
module = spy.Module.load_from_file(app.device, "nsc_nw_02_3layers_coopvec.slang")

# Load some materials.
image = spy.Tensor.load_from_image(app.device,
                                   "slangstars.png", linearize=False)

np.random.seed(0)

class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs
        self.layout = spy.CoopVecMatrixLayout.training_optimal

        # Create initial values of biases and weights
        weights_np = np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype(np.float16)
        biases_np = np.zeros(outputs).astype(np.float16)

        # Convert weights into coopvec layout for training
        desc = app.device.coopvec_create_matrix_desc(outputs, inputs, self.layout, spy.DataType.float16, 0)
        weight_count = desc.size // 2 # sizeof(half)
        params_np = np.zeros((weight_count, ), dtype=np.float16)
        app.device.coopvec_convert_matrix_host(weights_np, params_np, dst_layout=self.layout)

        # Create bias and weight tensors
        self.biases = spy.Tensor.zeros(app.device, (outputs, ), dtype='half')
        self.weights = spy.Tensor.zeros(app.device, (weight_count, ), dtype='half')
        self.biases.copy_from_numpy(biases_np)
        self.weights.copy_from_numpy(params_np)

        # Gradients for the biases and weights.
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)

        # Moment buffers for Adam optimizer.
        self.m_biases = spy.Tensor.zeros(app.device, self.biases.shape, 'float')
        self.m_weights = spy.Tensor.zeros(app.device, self.weights.shape, 'float')
        self.v_biases = spy.Tensor.zeros_like(self.m_biases)
        self.v_weights = spy.Tensor.zeros_like(self.m_weights)

        self.set_data({
            'biases': self.biases.storage,
            'weights': self.weights.storage,
            'biasGrads': self.biases_grad.storage,
            'weightGrads': self.weights_grad.storage,
            '_type': f"NetworkParameters<{inputs},{outputs}>"
        })

    # Calls the Slang 'optimize' function for biases and weights
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimize1(self.biases, self.biases_grad, self.m_biases, self.v_biases, learning_rate, optimize_counter)
        module.optimize1(self.weights, self.weights_grad, self.m_weights, self.v_weights, learning_rate, optimize_counter)

    def serialize(self):
        params_np = self.weights.to_numpy()
        weights_np = np.zeros((self.outputs, self.inputs), dtype=np.float16)
        app.device.coopvec_convert_matrix_host(params_np, weights_np, src_layout=self.layout)

        biases_np = self.biases.to_numpy()

        return {
            'num_inputs': self.inputs,
            'num_outputs': self.outputs,
            'weights': weights_np.flatten().tolist(),
            'biases': biases_np.tolist()
        }

class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.layer0 = NetworkParameters(16,32)
        self.layer1 = NetworkParameters(32,32)
        self.layer2 = NetworkParameters(32,3)

    # Calls the Slang 'optimize' function for the layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)

    def serialize(self):
        return {
            'layers': [
                self.layer0.serialize(),
                self.layer1.serialize(),
                self.layer2.serialize()
            ]
        }

if spy.Feature.cooperative_vector not in module.device.features:
    raise RuntimeError("Device does not support cooperative vector API")

network = Network()

optimize_counter = 0

while app.process_events():

    # Blit tensor to screen.
    offset = 0
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset,0), tonemap=False, bilinear=True)
    offset += 512 + 10
    res = spy.int2(256,256)

    lr_output = spy.Tensor.empty_like(image)
    module.render(pixel = spy.call_id(),
                  resolution = res,
                  network = network,
                  _result = lr_output)

    # Blit tensor to screen.
    app.blit(lr_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)
    offset += 512 + 10

    # Loss between downsampled output and quarter res rendered output.
    loss_output = spy.Tensor.empty_like(image)
    module.loss(pixel = spy.call_id(),
                  resolution = res,
                  network = network,
                  reference = image,
                  _result = loss_output)

    # Blit tensor to screen.
    app.blit(loss_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)
    offset += 512 + 10

    learning_rate = 0.001

    for i in range(50):
        # Loss between downsampled output and quarter res rendered output.
        module.calculate_grads(
            seed = spy.wang_hash(seed=optimize_counter, warmup=2),
            pixel = spy.call_id(),
            resolution = res,
            reference = image,
            network = network)
        optimize_counter += 1

        network.optimize(learning_rate, optimize_counter)

    print("Loss:", np.sum(np.abs(loss_output.to_numpy())))

    # Present the window.
    app.present()

open('weights.json', 'w').write(json.dumps(network.serialize(), indent=4))


