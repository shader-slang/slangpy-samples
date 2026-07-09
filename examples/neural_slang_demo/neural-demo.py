# SPDX-License-Identifier: Apache-2.0
from app import App
import slangpy as spy
import numpy as np
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--vector-type", choices=["inline", "wave"], default="inline")
parser.add_argument(
    "--device-type", choices=["automatic", "vulkan", "cuda", "metal"], default="automatic"
)
parser.add_argument(
    "--iterations",
    type=int,
    default=0,
    help="Run N training iterations headless (no window) and exit",
)
args = parser.parse_args()


def resolve_device_type(name: str) -> spy.DeviceType:
    if name == "automatic":
        return spy.DeviceType.metal if sys.platform == "darwin" else spy.DeviceType.vulkan
    return getattr(spy.DeviceType, name)


device_type = resolve_device_type(args.device_type)

TARGET_ENUMS = {
    spy.DeviceType.vulkan: "TargetEnum.SPIR_V",
    spy.DeviceType.cuda: "TargetEnum.CUDA",
    spy.DeviceType.metal: "TargetEnum.Metal",
}
defines: dict[str, str] = {}
if args.vector_type == "wave":
    # WaveTangledVector needs the compile target for shared memory sizing.
    defines["NEURAL_DEMO_WAVE"] = "1"
    defines["NEURAL_DEMO_TARGET"] = TARGET_ENUMS[device_type]

headless = args.iterations > 0
if headless:
    app = None
    device = spy.Device(
        type=device_type,
        compiler_options={
            "include_paths": [spy.SHADER_PATH, Path(__file__).parent],
            "defines": defines,
            "enable_experimental_features": True,
        },
    )
else:
    app = App(
        width=512 * 3 + 10 * 2,
        height=512,
        title="neural.slang demo",
        device_type=device_type,
        defines=defines,
    )
    device = app.device

# The link module provides the concrete vector types (InlineVector or
# WaveTangledVector) for the extern struct declarations in neural-demo.slang.
link_module = device.load_module(
    "neural-demo-wave.slang" if args.vector_type == "wave" else "neural-demo-inline.slang"
)
module = spy.Module.load_from_file(device, "neural-demo.slang", link=[link_module])
image = spy.Tensor.load_from_image(
    device, Path(__file__).parent.joinpath("slangstars.png"), linearize=True
)


def param_buffer_ref(tensor: spy.Tensor):
    return tensor.storage.device_address


class LatentTexture(spy.InstanceList):
    def __init__(self, width: int, height: int, num_latents: int):
        super().__init__(module[f"LatentTexture<{num_latents}>"])
        initial = np.random.uniform(0.0, 1.0, (height, width, num_latents)).astype("float32")
        self._tex = spy.Tensor.from_numpy(device, initial)
        self._tex_grad = spy.Tensor.zeros_like(self._tex)
        self.texture, self.texture_grads = self._tex, self._tex_grad
        self._m, self._v = spy.Tensor.zeros_like(self._tex), spy.Tensor.zeros_like(self._tex)

    def optimize(self, lr: float, it: int):
        module.optimizer_step(self._tex, self._tex_grad, self._m, self._v, lr, it)


LAYERS = [(4, 32), (32, 32), (32, 3)]


def xavier_init(layers, optimal: bool):
    # Parameters are initialized directly in the layout the layers read from
    # (LinearLayout: tightly packed row-major weights + bias per layer;
    # OptimalLayout: 16-padded per-layer blocks). Since training starts from random
    # values, the position of each value within a weight block doesn't matter:
    # each layer's weight block gets Xavier-scaled noise and its bias block zeros.
    # Values landing in layout padding are inert.
    def pad(x):
        return (x + 15) // 16 * 16 if optimal else x

    params = []
    for inp, out in layers:
        scale = np.sqrt(6.0 / (inp + out))
        params.extend([np.random.uniform(-scale, scale, pad(out) * pad(inp)), np.zeros(pad(out))])
    return np.concatenate(params).astype("float32")


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        params_np = xavier_init(LAYERS, optimal=args.vector_type == "wave")

        # Cross-check the host-computed parameter buffer size against the neural
        # module, via Slang reflection (no GPU roundtrip needed).
        #
        # The neural module's NetworkParameterLayoutConverter<T, BiasMask, D0, D1,
        # ...> describes a whole network's parameter block and exposes its sizes as
        # static constants: ElementCountLogical for the tightly packed row-major
        # layout (LinearLayout) and ElementCountPhysical for the 16-padded tiled
        # layout (OptimalLayout). Specialize it for this network: the dimensions
        # 4, 32, 32, 3, and one bias-mask bit per layer (0b111 = every layer has a
        # bias).
        dims = ", ".join(str(d) for d in [LAYERS[0][0]] + [out for _, out in LAYERS])
        bias_mask = (1 << len(LAYERS)) - 1
        converter = f"NetworkParameterLayoutConverter<float, {bias_mask}, {dims}>"
        member = "ElementCountPhysical" if args.vector_type == "wave" else "ElementCountLogical"
        # Reflection cannot read a static constant's value directly, but
        # find_type_by_name() evaluates full type expressions: wrapping the
        # constant as an array bound and reading back the array's element count
        # yields its value.
        size = module.layout.program_layout.find_type_by_name(
            f"float[{converter}.{member}]"
        ).element_count
        assert params_np.size == size
        self._params = spy.Tensor.from_numpy(device, params_np)
        self._params_grad = spy.Tensor.zeros_like(self._params)
        self._m, self._v = spy.Tensor.zeros_like(self._params), spy.Tensor.zeros_like(self._params)
        self.params = param_buffer_ref(self._params)
        self.params_grad = param_buffer_ref(self._params_grad)
        self.latent_texture = LatentTexture(32, 32, 4)

    def optimize(self, lr: float, it: int):
        module.optimizer_step(self._params, self._params_grad, self._m, self._v, lr, it)
        self.latent_texture.optimize(lr, it)


network = Network()
iteration = 0
res, batch_size, lr = spy.int2(256, 256), (64, 64), 0.001


def train_step():
    global iteration
    module.calculate_grads(
        seed=spy.wang_hash(seed=iteration, warmup=2),
        batch_index=spy.grid(batch_size),
        batch_size=spy.int2(batch_size),
        reference=image,
        network=network,
    )
    iteration += 1
    network.optimize(lr, iteration)


def compute_loss() -> float:
    loss_out = spy.Tensor.empty_like(image)
    module.show_loss(
        pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_out
    )
    return float(np.mean(loss_out.to_numpy()))


print("Compiling shaders...")

if headless:
    initial_loss = compute_loss()
    print(f"Initial loss: {initial_loss:.5f}")
    for _ in range(args.iterations):
        train_step()
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, loss: {compute_loss():.5f}")
    final_loss = compute_loss()
    print(f"Final loss: {final_loss:.5f}")
    if not final_loss < initial_loss:
        print("ERROR: loss did not decrease")
        sys.exit(1)
else:
    assert app is not None
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
            train_step()

        print(f"Loss: {np.mean(loss_out.to_numpy()):.5f}")
        app.present()
