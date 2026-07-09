# Neural Demo - Using neural.slang

![Screenshot](screenshot.png)

This demo showcases how to use Slang's neural.slang standard module (imported as `slang.neural`) to build a neural network for image reconstruction. The network learns to map UV coordinates to RGB colors, reconstructing a reference image through gradient-based optimization.
This is a re-creation of the texture example in the https://github.com/shader-slang/neural-shading-s25 course.

## neural.slang Types Used

| Type | Description |
|------|-------------|
| `IVector<T>` | Interface for all vector types |
| `InlineVector<T, N>` | Fixed-size vector type with compile-time `.Size` constant |
| `WaveTangledVector<T, ShMemSize, N, SubgroupSize, SubgroupCount>` | Cooperative-matrix-accelerated vector (requires GPU support) |
| `IPointerLikeAddress<T>` | Pointer-like parameter storage abstraction |
| `PointerAddress<T>` | `IPointerLikeAddress` over a raw device pointer |
| `FFLayer<T, InVec, OutVec, Layout, Activation, HasBias>` | Feed-forward neural network layer |
| `LinearLayout` / `OptimalLayout` | Parameter storage layouts (row-major / tiled for cooperative matrices) |
| `LeakyReLU<T>` | Leaky ReLU activation with configurable alpha (default 0.01) |
| `ExpActivation<T>` | Exponential activation: `exp(x)` |
| `SharedMemorySize` | Shared memory pool sizing for `WaveTangledVector` cooperative operations |

## Comparison with the Hand-Written Version

Lines of code compared to the hand-written course example
(`network/step_05_latent_texture` in neural-shading-s25), excluding comments and
blank lines:

| | Hand-written original | neural.slang (this demo) |
|---|-----------------------|--------------------------|
| Network definition + forward pass | ~68 | ~10 |

The `NetworkParameters` struct (manual weight/bias accessors with custom backward
derivatives and a hand-written matrix multiply) and the manually unrolled
activation loops in `Network.eval` collapse into three `FFLayer` typealiases and a
five-line `mlp_forward`. On top of that, the original supports a single vector
flavor, while this demo selects between `InlineVector` and the
cooperative-matrix-accelerated `WaveTangledVector` at link time with no changes to
the network code.

## Network Structure

A small MLP (4 -> 32 -> 32 -> 3) fed by a trainable 32x32x4 latent texture:

```slang
typealias Layer0 = FFLayer<float, InputVec, HiddenVec, Layout, LeakyReLU<float>, true>;
typealias Layer1 = FFLayer<float, HiddenVec, HiddenVec, Layout, LeakyReLU<float>, true>;
typealias Layer2 = FFLayer<float, HiddenVec, OutputVec, Layout, ExpActivation<float>, true>;

[Differentiable]
float3 mlp_forward(Address params, InputVec input)
{
    let h0 = Layer0(LeakyReLU<float>(LEAKY_RELU_SLOPE)).eval<Address>(input, params);
    let h1 = Layer1(LeakyReLU<float>(LEAKY_RELU_SLOPE)).eval<Address>(h0, params.getOffset(Layer0.nextOffset(0)));
    let output = Layer2(ExpActivation<float>()).eval<Address>(h1, params.getOffset(Layer1.nextOffset(Layer0.nextOffset(0))));
    return float3(output[0], output[1], output[2]);
}
```

The whole network reads from a single `IPointerLikeAddress`; per-layer blocks
(weights followed by bias) are derived inline with the differentiable `getOffset`
and `FFLayer.nextOffset()`.

## Training

The full loss is differentiable end to end: the latent texture is sampled with a
bilinear `LatentTexture.sample` (texture gradients accumulate through a custom
derivative on `getLatent` into an `AtomicTensor`), the network runs `mlp_forward`,
and the gamma-corrected L2 loss is computed in `loss`. `calculate_grads` invokes

```slang
bwd_diff(loss)(
    DifferentialPtrPair<Address>(Address(network.params), Address(network.params_grad)),
    network.latent_texture, pixel, resolution, ref_color, float3(1.0f));
```

so parameter gradients accumulate through the `DifferentialPtrPair` (primal =
parameters, differential = gradients) and latent texture gradients through the
custom derivative.

## Link-Time Vector Type Selection

The three vector types are declared as `extern struct` in `neural-demo.slang`,
making them link-time types:

```slang
extern struct InputVec : IVector<float>;
extern struct HiddenVec : IVector<float>;
extern struct OutputVec : IVector<float>;
```

Separate modules provide the concrete implementations, and the Python script links
the appropriate one based on the `--vector-type` command line flag:

**`neural-demo-inline.slang`** (default):
```slang
export struct InputVec : IVector<float> = InlineVector<float, 4>;
export struct HiddenVec : IVector<float> = InlineVector<float, 32>;
export struct OutputVec : IVector<float> = InlineVector<float, 3>;
```
plain per-thread vectors with the row-major `LinearLayout` parameter layout.

**`neural-demo-wave.slang`** (accelerated):
```slang
export struct InputVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 4, SubgroupSize, SubgroupCount>;
export struct HiddenVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 32, SubgroupSize, SubgroupCount>;
export struct OutputVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 3, SubgroupSize, SubgroupCount>;
```
cooperative-matrix accelerated vectors with a shared memory pool sized via
`SharedMemorySize`. These require the tiled `OptimalLayout` parameter layout
(selected in `neural-demo.slang` via the `NEURAL_DEMO_WAVE` define), which is also
the only layout supported on Metal.

Because `FFLayer.eval()` is generic over `IVector<T>` and the interface's element
accessors are `[Differentiable]` (requires
https://github.com/shader-slang/slang/pull/12026), the network code in
`neural-demo.slang` is completely agnostic to the underlying vector implementation:
it constructs `InputVec` and extracts output components directly through
`operator[]`.

Because `FFLayer.eval()` is generic over `IVector<T>`, the network code in
`neural-demo.slang` is identical for both variants.

## Parameter Layouts

Parameters live in one flat buffer, one contiguous block per layer (weights followed
by bias), in whatever layout the selected `Layout` reads: tightly packed row-major
for `LinearLayout`, 16-padded tiled blocks for `OptimalLayout`. Since this sample
always trains from scratch, the host initializes the buffer directly in that native
layout - each layer's weight block gets Xavier-scaled random values and its bias
block zeros; where a value lands inside a random weight block doesn't matter, and
values in layout padding are inert. No layout conversion is needed. Gradients
accumulate in the same layout, so the Adam optimizer runs elementwise over the
buffer.

### Querying the buffer size via reflection

The neural module's `NetworkParameterLayoutConverter<T, BiasMask, D0, D1, ...>`
describes a whole network's parameter block and exposes its sizes as static
constants (`ElementCountLogical` / `ElementCountPhysical`, and `BytesLogical` /
`BytesPhysical`). The Python driver uses these to validate its host-computed
buffer size through Slang reflection, without any GPU roundtrip:

```python
converter = "NetworkParameterLayoutConverter<float, 7, 4, 32, 32, 3>"
size = module.layout.program_layout.find_type_by_name(
    f"float[{converter}.ElementCountPhysical]"
).element_count
```

Reflection cannot read a static constant's value directly from Python, but
`find_type_by_name()` evaluates full type expressions - wrapping the constant as
an array bound and reading back `element_count` yields its value.

(For deploying *pretrained* row-major weights into `OptimalLayout`, the same
converter type provides `toOptimalLayout()` to convert layouts on the GPU; this
sample trains from scratch and doesn't need it.)

## Platform Support

| Vector type | Vulkan | CUDA | Metal |
|-------------|--------|------|-------|
| `InlineVector` (LinearLayout) | yes | yes | yes |
| `WaveTangledVector` (OptimalLayout) | yes | yes | yes |

Parameter buffers are passed as raw device pointers (`PointerAddress<float>`),
which work on all three backends (Vulkan uses buffer device addresses).

## File Structure

| File | Description |
|------|-------------|
| `neural-demo.slang` | Main network: MLP forward pass, loss, training |
| `neural-demo-inline.slang` | Link module: all vectors as `InlineVector` |
| `neural-demo-wave.slang` | Link module: all vectors as `WaveTangledVector` with shared memory pool |
| `neural-demo.py` | Python driver: device setup, vector type selection, training loop |
| `app.py` | Windowing and display utilities |
| `app.slang` | Blit/tonemap helper shaders |

## Running the Demo

```bash
cd samples/examples/neural_slang_demo

# Default mode (InlineVector); device defaults to Vulkan (Metal on macOS)
python neural-demo.py

# Accelerated mode (WaveTangledVector, requires cooperative matrix GPU support)
python neural-demo.py --vector-type wave

# Select the GPU backend explicitly
python neural-demo.py --device-type cuda
python neural-demo.py --vector-type wave --device-type metal

# Headless smoke test: run N training iterations without a window and verify
# the loss decreases
python neural-demo.py --vector-type wave --device-type vulkan --iterations 300
```

The windowed demo displays three panels:
1. **Reference image** - Target to reconstruct
2. **Network output** - Current reconstruction using FFLayer-based network
3. **Loss visualization** - Per-pixel error

Loss values are printed to console and should decrease over time as the network learns.
