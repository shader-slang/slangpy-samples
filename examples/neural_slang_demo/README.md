# Neural Demo - Using neural.slang

![Screenshot](screenshot.png)

This demo showcases how to use Slang's `neural.slang` standard module to build a neural network for image reconstruction. The network learns to map UV coordinates to RGB colors, reconstructing a reference image through gradient-based optimization.
This is a re-creation of the texture example in the https://github.com/shader-slang/neural-shading-s25 course.

## neural.slang Types Used

| Type | Description |
|------|-------------|
| `InlineVector<T, N>` | Fixed-size vector type with compile-time `.Size` constant |
| `WaveTangledVector<T, Pool, N, WaveSize>` | Cooperative-matrix-accelerated vector (requires GPU support) |
| `StructuredBufferStorage<T>` | GPU buffer storage implementing `IStorage<T>` interface |
| `FFLayer<T, InVec, OutVec, Storage, Layout, Activation, HasBias>` | Feed-forward neural network layer |
| `LinearLayout` | Storage layout for standard linear (row-major) weight packing |
| `LeakyReLU<T>` | Leaky ReLU activation with configurable alpha (default 0.01) |
| `ExpActivation<T>` | Exponential activation: `exp(x)` |

## Before/After Comparison

This section shows the comparison between the original code and the code with the neural.slang APIs

### Lines of Code

Approximate lines of code comparison apart from comments.

| File Type | Original | neural.slang |
|-----------|----------|--------------|
| Slang | 171 | 105 |
| Python | 103 | 68 |

### Vector Types

| Before (Manual) | After (neural.slang) |
|-----------------|---------------------|
| `float[4]` / `float4` | `InlineVector<float, 4>` |
| `float[32]` | `InlineVector<float, 32>` |
| `float[3]` / `float3` | `InlineVector<float, 3>` |
| Manual size tracking | `InlineVector.Size` compile-time constant |

### Parameter Storage

| Before (Manual) | After (neural.slang) |
|-----------------|---------------------|
| Separate weight/bias buffers | `StructuredBufferStorage<T>` wrapper |
| Manual offset calculation | `Storage.getOffset()` method |
| Manual parameter count | `FFLayer.ParameterCount` constant |

### Layer Forward Pass

| Before (Manual) | After (neural.slang) |
|-----------------|---------------------|
| Manual matrix multiply | `FFLayer.eval()` using `linearTransform` |
| Explicit loops | Optimized internal implementation |
| Manual bias addition | Handled by `FFLayer` |

**Before:**
```slang
// Single layer forward pass
[Differentiable]
float[Outputs] forward(float[Inputs] x)
{
    float[Outputs] y;
    [MaxIters(Outputs)]
    for (int row = 0; row < Outputs; ++row)
    {
        var sum = get_bias(row);
        [ForceUnroll]
        for (int col = 0; col < Inputs; ++col)
            sum += get_weight(row, col) * x[col];
        y[row] = sum;
    }
    return y;
}

// Network evaluation with manual activation loops
[Differentiable]
float3 eval(no_diff float2 uv)
{
    float encoded_inputs[NumLatents] = latent_texture.sample(uv);

    float output0[32] = layer0.forward(encoded_inputs);
    [ForceUnroll]
    for (int i = 0; i < 32; ++i)
        output0[i] = leakyReLU(output0[i]);
    float output1[32] = layer1.forward(output0);
    [ForceUnroll]
    for (int i = 0; i < 32; ++i)
        output1[i] = leakyReLU(output1[i]);
    float output2[3] = layer2.forward(output1);
    [ForceUnroll]
    for (int i = 0; i < 3; ++i)
        output2[i] = exp(output2[i]);
    return float3(output2[0], output2[1], output2[2]);
}
```

**After:**
```slang
// Type aliases using FFLayer with Layout and Activation parameters
typealias Storage = StructuredBufferStorage<float>;
typealias Layer0 = FFLayer<float, InputVec, HiddenVec, Storage, LinearLayout, LeakyReLU<float>>;
typealias Layer1 = FFLayer<float, HiddenVec, HiddenVec, Storage, LinearLayout, LeakyReLU<float>>;
typealias Layer2 = FFLayer<float, HiddenVec, OutputVec, Storage, LinearLayout, ExpActivation<float>>;

// Network evaluation with FFLayer and built-in activations
[Differentiable]
OutputVec mlp_forward(Storage storage, InputVec input)
{
    uint addr = 0u;
    let h0 = Layer0(addr, addr + INPUT_DIM * HIDDEN_DIM, LeakyReLU<float>(LEAKY_RELU_SLOPE)).eval<Storage>(storage, input);
    addr = Layer0.nextAddress(addr);
    let h1 = Layer1(addr, addr + HIDDEN_DIM * HIDDEN_DIM, LeakyReLU<float>(LEAKY_RELU_SLOPE)).eval<Storage>(storage, h0);
    addr = Layer1.nextAddress(addr);
    return Layer2(addr, addr + HIDDEN_DIM * OUTPUT_DIM, ExpActivation<float>()).eval<Storage>(storage, h1);
}
```

### Network Definition

| Before (Manual) | After (neural.slang) |
|-----------------|---------------------|
| Custom struct with manual layout | Type aliases for layers |
| Hardcoded dimensions | Dimensions from vector types |
| Manual weight indexing | Automatic address calculation |

## Link-Time Type Selection

The hidden vector type (`HiddenVec`) is declared as an `extern struct` in `neural-demo.slang`, making it a link-time type. Separate `.slang` files provide the concrete implementation, and the Python script links the appropriate one based on a command-line flag:

- **`neural-demo-inline.slang`:** `HiddenVec = InlineVector<float, 32>` — standard fixed-size vector (default)
- **`neural-demo-wave.slang`:** `HiddenVec = WaveTangledVector<float, ShMemPool, 32, 32>` — cooperative-matrix-accelerated vector

Only `HiddenVec` (32-dim) is a link-time type because `FFLayer.eval()` is generic over vector types, so an `InlineVector` input naturally produces a `WaveTangledVector` hidden output. Input (4-dim) and output (3-dim) vectors are too small to benefit from cooperative matrix acceleration.

## Running the Demo

```bash
cd slangpy-samples/examples/neural_slang_demo

# Default mode (InlineVector)
python neural-demo.py

# Accelerated mode (WaveTangledVector, requires cooperative matrix GPU support)
python neural-demo.py --vector-type wave
```

The demo displays three panels:
1. **Reference image** - Target to reconstruct
2. **Network output** - Current reconstruction using FFLayer-based network
3. **Loss visualization** - Per-pixel error

Loss values are printed to console and should decrease over time as the network learns.
