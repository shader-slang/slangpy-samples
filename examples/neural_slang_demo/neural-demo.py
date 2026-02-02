# SPDX-License-Identifier: Apache-2.0
from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

app = App(
    width=512 * 3 + 10 * 2,
    height=512,
    title="neural.slang demo",
    device_type=spy.DeviceType.vulkan,
)
module = spy.Module.load_from_file(app.device, "neural-demo.slang")
image = spy.Tensor.load_from_image(
    app.device, Path(__file__).parent.joinpath("slangstars.png"), linearize=True
)


class LatentTexture(spy.InstanceList):
    def __init__(self, width: int, height: int, num_latents: int):
        super().__init__(module[f"LatentTexture<{num_latents}>"])
        initial = np.random.uniform(0.0, 1.0, (height, width, num_latents)).astype("float32")
        self._tex = spy.Tensor.from_numpy(app.device, initial)
        self._tex_grad = spy.Tensor.zeros_like(self._tex)
        self.texture, self.texture_grads = self._tex, self._tex_grad
        self._m, self._v = spy.Tensor.zeros_like(self._tex), spy.Tensor.zeros_like(self._tex)

    def optimize(self, lr: float, it: int):
        module.optimizer_step(self._tex, self._tex_grad, self._m, self._v, lr, it)


def xavier_init(layers):
    params = []
    for inp, out in layers:
        scale = np.sqrt(6.0 / (inp + out))
        params.extend([np.random.uniform(-scale, scale, out * inp), np.zeros(out)])
    return np.concatenate(params).astype("float32")


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        params_np = xavier_init([(4, 32), (32, 32), (32, 3)])
        self._params = spy.Tensor.from_numpy(app.device, params_np)
        self._params_grad = spy.Tensor.zeros_like(self._params)
        self._m, self._v = spy.Tensor.zeros_like(self._params), spy.Tensor.zeros_like(self._params)
        self.params, self.params_grad = self._params.storage, self._params_grad.storage
        self.latent_texture = LatentTexture(32, 32, 4)

    def optimize(self, lr: float, it: int):
        module.optimizer_step(self._params, self._params_grad, self._m, self._v, lr, it)
        self.latent_texture.optimize(lr, it)


network = Network()
iteration = 0
res, batch_size, lr = spy.int2(256, 256), (64, 64), 0.001

print("Compiling shaders...")

while app.process_events():
    app.blit(image, size=spy.int2(512), offset=spy.int2(0, 0), tonemap=False, bilinear=True)

    output = spy.Tensor.empty_like(image)
    module.render(pixel=spy.call_id(), resolution=res, network=network, _result=output)
    app.blit(output, size=spy.int2(512), offset=spy.int2(522, 0), tonemap=False, bilinear=True)

    loss_out = spy.Tensor.empty_like(image)
    module.show_loss(
        pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_out
    )
    app.blit(loss_out, size=spy.int2(512), offset=spy.int2(1044, 0), tonemap=False)

    for _ in range(20):
        module.calculate_grads(
            seed=spy.wang_hash(seed=iteration, warmup=2),
            batch_index=spy.grid(batch_size),
            batch_size=spy.int2(batch_size),
            reference=image,
            network=network,
        )
        iteration += 1
        network.optimize(lr, iteration)

    print(f"Loss: {np.mean(loss_out.to_numpy()):.5f}")
    app.present()
