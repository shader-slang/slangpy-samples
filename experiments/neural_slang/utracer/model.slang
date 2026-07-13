// SPDX-License-Identifier: Apache-2.0

// Shared neural material model used by both training and UTracer.
// Inline vectors use portable row-major parameters. Wave-tangled vectors use
// the cooperative-matrix-friendly tiled layout.

import slang.neural;

public static const int kNumLatents = 8;
public static const int kNumBsdfLayers = 2;
public static const int kBsdfLayerInputs = 14;
public static const int kEncoderInputs = kNumBsdfLayers * kBsdfLayerInputs + 1;
public static const int kEncoderHidden = 64;
public static const int kDecoderInputs = kNumLatents + 6;
public static const int kDecoderHidden = 32;

public typealias NeuralEncoderParameterConverter =
    NetworkParameterLayoutConverter<
        float,
        0b1111,
        kEncoderInputs,
        kEncoderHidden,
        kEncoderHidden,
        kEncoderHidden,
        kNumLatents>;

public typealias NeuralDecoderParameterConverter =
    NetworkParameterLayoutConverter<
        float,
        0b111,
        kDecoderInputs,
        kDecoderHidden,
        kDecoderHidden,
        3>;

#if NEURAL_VECTOR_WAVE
public typealias NeuralLayout = OptimalLayout;
#else
public typealias NeuralLayout = LinearLayout;
#endif

public typealias NeuralAddress = PointerAddress<float>;

// SlangPy generates 32-thread compute groups for ordinary vectorized calls.
// Keeping one subgroup per workgroup makes the WaveTangled specialization match
// those generated entry points.
static const int kSubgroupSize = 32;
static const int kSubgroupCount = 1;

#if NEURAL_EXECUTION_INFERENCE
#define NEURAL_EXECUTION_MODE ExecutionMode.Inference
#else
#define NEURAL_EXECUTION_MODE ExecutionMode.Training
#endif

#if NEURAL_TARGET_CUDA
typealias NeuralSharedMemory = SharedMemorySize<
    float,
    TargetEnum.CUDA,
    NEURAL_EXECUTION_MODE,
    kSubgroupSize,
    kSubgroupCount,
#if !NEURAL_EXECUTION_INFERENCE
    kEncoderInputs,
    kEncoderHidden,
    kEncoderHidden,
    kEncoderHidden,
    kNumLatents,
#endif
    kDecoderInputs,
    kDecoderHidden,
    kDecoderHidden,
    3>;
#elif NEURAL_TARGET_METAL
typealias NeuralSharedMemory = SharedMemorySize<
    float,
    TargetEnum.Metal,
    NEURAL_EXECUTION_MODE,
    kSubgroupSize,
    kSubgroupCount,
#if !NEURAL_EXECUTION_INFERENCE
    kEncoderInputs,
    kEncoderHidden,
    kEncoderHidden,
    kEncoderHidden,
    kNumLatents,
#endif
    kDecoderInputs,
    kDecoderHidden,
    kDecoderHidden,
    3>;
#else
typealias NeuralSharedMemory = SharedMemorySize<
    float,
    TargetEnum.SPIR_V,
    NEURAL_EXECUTION_MODE,
    kSubgroupSize,
    kSubgroupCount,
#if !NEURAL_EXECUTION_INFERENCE
    kEncoderInputs,
    kEncoderHidden,
    kEncoderHidden,
    kEncoderHidden,
    kNumLatents,
#endif
    kDecoderInputs,
    kDecoderHidden,
    kDecoderHidden,
    3>;
#endif

#if NEURAL_VECTOR_WAVE
public typealias NeuralEncoderInputVector = WaveTangledVector<
    float, NeuralSharedMemory, kEncoderInputs, kSubgroupSize, kSubgroupCount>;
public typealias NeuralEncoderHiddenVector = WaveTangledVector<
    float, NeuralSharedMemory, kEncoderHidden, kSubgroupSize, kSubgroupCount>;
public typealias NeuralLatentVector = WaveTangledVector<
    float, NeuralSharedMemory, kNumLatents, kSubgroupSize, kSubgroupCount>;
public typealias NeuralDecoderInputVector = WaveTangledVector<
    float, NeuralSharedMemory, kDecoderInputs, kSubgroupSize, kSubgroupCount>;
public typealias NeuralDecoderHiddenVector = WaveTangledVector<
    float, NeuralSharedMemory, kDecoderHidden, kSubgroupSize, kSubgroupCount>;
public typealias NeuralOutputVector = WaveTangledVector<
    float, NeuralSharedMemory, 3, kSubgroupSize, kSubgroupCount>;
#else
public typealias NeuralEncoderInputVector = InlineVector<float, kEncoderInputs>;
public typealias NeuralEncoderHiddenVector = InlineVector<float, kEncoderHidden>;
public typealias NeuralLatentVector = InlineVector<float, kNumLatents>;
public typealias NeuralDecoderInputVector = InlineVector<float, kDecoderInputs>;
public typealias NeuralDecoderHiddenVector = InlineVector<float, kDecoderHidden>;
public typealias NeuralOutputVector = InlineVector<float, 3>;
#endif

/// Encoder implementation abstracted over IVector-compatible vector types.
public struct Encoder<InputVector, HiddenVector, LatentVector>
    where InputVector : IVector<float>
    where HiddenVector : IVector<float>
    where LatentVector : IVector<float>
{
    public typealias Layer0 = FFLayer<
        float, InputVector, HiddenVector, NeuralLayout, LeakyReLU<float>>;
    public typealias Layer1 = FFLayer<
        float, HiddenVector, HiddenVector, NeuralLayout, LeakyReLU<float>>;
    public typealias Layer2 = FFLayer<
        float, HiddenVector, HiddenVector, NeuralLayout, LeakyReLU<float>>;
    public typealias Layer3 = FFLayer<
        float, HiddenVector, LatentVector, NeuralLayout, IdentityActivation<float>>;

    public static const int Layer0Offset = 0;
    public static const int Layer1Offset = Layer0Offset + Layer0.StorageSize;
    public static const int Layer2Offset = Layer1Offset + Layer1.StorageSize;
    public static const int Layer3Offset = Layer2Offset + Layer2.StorageSize;
    public static const int ParameterCount = Layer3Offset + Layer3.StorageSize;

    public Layer0 layer0;
    public Layer1 layer1;
    public Layer2 layer2;
    public Layer3 layer3;

    public __init()
    {
        LeakyReLU<float> leaky = LeakyReLU<float>(0.01);
        layer0 = Layer0(leaky);
        layer1 = Layer1(leaky);
        layer2 = Layer2(leaky);
        layer3 = Layer3();
    }

    [Differentiable]
    public LatentVector eval<A>(InputVector input, A parameters)
        where A : IPointerLikeAddress<float>
        where A.Differential : IPointerLikeAddress<float>
    {
        HiddenVector h0 = layer0.eval<A>(input, parameters.getOffset(Layer0Offset));
        HiddenVector h1 = layer1.eval<A>(h0, parameters.getOffset(Layer1Offset));
        HiddenVector h2 = layer2.eval<A>(h1, parameters.getOffset(Layer2Offset));
        return layer3.eval<A>(h2, parameters.getOffset(Layer3Offset));
    }
};

[Differentiable]
static void parameterizeDirections(
    float3 wi,
    float3 wo,
    out float3 wh,
    out float3 wd)
{
    wh = normalize(wi + wo);
    float3 v = float3(wh.y, -wh.x, 0.0f);
    wd = wi * wh.z + cross(v, wi) + dot(v, wi) * v / (1.0f + wh.z);
}

/// Decoder implementation abstracted over IVector-compatible vector types.
public struct Decoder<LatentVector, InputVector, HiddenVector, OutputVector>
    where LatentVector : IVector<float>
    where InputVector : IVector<float>
    where HiddenVector : IVector<float>
    where OutputVector : IVector<float>
{
    public typealias Layer0 = FFLayer<
        float, InputVector, HiddenVector, NeuralLayout, LeakyReLU<float>>;
    public typealias Layer1 = FFLayer<
        float, HiddenVector, HiddenVector, NeuralLayout, LeakyReLU<float>>;
    public typealias Layer2 = FFLayer<
        float, HiddenVector, OutputVector, NeuralLayout, ExpActivation<float>>;

    public static const int Layer0Offset = 0;
    public static const int Layer1Offset = Layer0Offset + Layer0.StorageSize;
    public static const int Layer2Offset = Layer1Offset + Layer1.StorageSize;
    public static const int ParameterCount = Layer2Offset + Layer2.StorageSize;

    public Layer0 layer0;
    public Layer1 layer1;
    public Layer2 layer2;

    public __init()
    {
        LeakyReLU<float> leaky = LeakyReLU<float>(0.01);
        layer0 = Layer0(leaky);
        layer1 = Layer1(leaky);
        layer2 = Layer2();
    }

    [Differentiable]
    public float3 eval<A>(
        LatentVector latents,
        float3 wi,
        float3 wo,
        A parameters)
        where A : IPointerLikeAddress<float>
        where A.Differential : IPointerLikeAddress<float>
    {
        float3 wh, wd;
        parameterizeDirections(wi, wo, wh, wd);

        InputVector input;
        [ForceUnroll]
        for (int i = 0; i < kNumLatents; ++i)
            input[i] = latents[i];
        input[kNumLatents + 0] = wh.x;
        input[kNumLatents + 1] = wh.y;
        input[kNumLatents + 2] = wh.z;
        input[kNumLatents + 3] = wd.x;
        input[kNumLatents + 4] = wd.y;
        input[kNumLatents + 5] = wd.z;

        return evalInput<A>(input, parameters);
    }

    [Differentiable]
    public float3 evalInput<A>(InputVector input, A parameters)
        where A : IPointerLikeAddress<float>
        where A.Differential : IPointerLikeAddress<float>
    {
        HiddenVector h0 = layer0.eval<A>(input, parameters.getOffset(Layer0Offset));
        HiddenVector h1 = layer1.eval<A>(h0, parameters.getOffset(Layer1Offset));
        OutputVector output = layer2.eval<A>(h1, parameters.getOffset(Layer2Offset));
        return float3(output[0], output[1], output[2]);
    }
};

/// Complete encoder/decoder network. The same type and parameter layout are
/// shared by training and baked-latent rendering.
public struct Network<
    EncoderInputVector,
    EncoderHiddenVector,
    LatentVector,
    DecoderInputVector,
    DecoderHiddenVector,
    OutputVector>
    where EncoderInputVector : IVector<float>
    where EncoderHiddenVector : IVector<float>
    where LatentVector : IVector<float>
    where DecoderInputVector : IVector<float>
    where DecoderHiddenVector : IVector<float>
    where OutputVector : IVector<float>
{
    public typealias EncoderType = Encoder<
        EncoderInputVector, EncoderHiddenVector, LatentVector>;
    public typealias DecoderType = Decoder<
        LatentVector, DecoderInputVector, DecoderHiddenVector, OutputVector>;

    public static const int EncoderParameterCount = EncoderType.ParameterCount;
    public static const int DecoderParameterOffset = EncoderParameterCount;
    public static const int ParameterCount =
        EncoderParameterCount + DecoderType.ParameterCount;

    public EncoderType encoder;
    public DecoderType decoder;

    public __init()
    {
        encoder = EncoderType();
        decoder = DecoderType();
    }

    [Differentiable]
    public float3 eval<A>(
        float[kEncoderInputs] encoderInputs,
        float3 wi,
        float3 wo,
        A parameters)
        where A : IPointerLikeAddress<float>
        where A.Differential : IPointerLikeAddress<float>
    {
        EncoderInputVector input = EncoderInputVector(encoderInputs);
        return evalVector<A>(input, wi, wo, parameters);
    }

    [Differentiable]
    public float3 evalVector<A>(
        EncoderInputVector input,
        float3 wi,
        float3 wo,
        A parameters)
        where A : IPointerLikeAddress<float>
        where A.Differential : IPointerLikeAddress<float>
    {
        LatentVector latents = encoder.eval<A>(input, parameters);
        return evalBaked<A>(latents, wi, wo, parameters);
    }

    [Differentiable]
    public float3 evalBaked<A>(
        LatentVector latents,
        float3 wi,
        float3 wo,
        A parameters)
        where A : IPointerLikeAddress<float>
        where A.Differential : IPointerLikeAddress<float>
    {
        return decoder.eval<A>(
            latents, wi, wo, parameters.getOffset(DecoderParameterOffset));
    }
};

public typealias NeuralEncoder = Encoder<
    NeuralEncoderInputVector,
    NeuralEncoderHiddenVector,
    NeuralLatentVector>;

public typealias NeuralDecoder = Decoder<
    NeuralLatentVector,
    NeuralDecoderInputVector,
    NeuralDecoderHiddenVector,
    NeuralOutputVector>;

public typealias NeuralNetwork = Network<
    NeuralEncoderInputVector,
    NeuralEncoderHiddenVector,
    NeuralLatentVector,
    NeuralDecoderInputVector,
    NeuralDecoderHiddenVector,
    NeuralOutputVector>;

public static const int kEncoderParameterCount = NeuralNetwork.EncoderParameterCount;
public static const int kDecoderParameterOffset = NeuralNetwork.DecoderParameterOffset;
public static const int kNetworkParameterCount = NeuralNetwork.ParameterCount;
public static const int kPortableNetworkParameterCount =
    NeuralEncoderParameterConverter.ElementCountLogical +
    NeuralDecoderParameterConverter.ElementCountLogical;
public static const int kOptimalNetworkParameterCount =
    NeuralEncoderParameterConverter.ElementCountPhysical +
    NeuralDecoderParameterConverter.ElementCountPhysical;
