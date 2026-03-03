// SPDX-License-Identifier: Apache-2.0
import slangpy;
import neural;

static const int INPUT_DIM = 4;
static const int HIDDEN_DIM = 32;
static const int OUTPUT_DIM = 3;
static const float LEAKY_RELU_SLOPE = 0.01f;

extern struct InputVec : IVector<float>;
extern struct HiddenVec : IVector<float>;
extern struct OutputVec : IVector<float>;
[Differentiable] extern InputVec arrayToInputVec(float data[INPUT_DIM]);
[Differentiable] extern float3 outputToFloat3(OutputVec v);

typealias Address = BindlessAddress<float>;
typealias Layer0 = FFLayer<float, InputVec, HiddenVec, LinearLayout, LeakyReLU<float>, true>;
typealias Layer1 = FFLayer<float, HiddenVec, HiddenVec, LinearLayout, LeakyReLU<float>, true>;
typealias Layer2 = FFLayer<float, HiddenVec, OutputVec, LinearLayout, ExpActivation<float>, true>;

// WAR(https://github.com/shader-slang/slang/issues/10195): getOffset is not yet differentiable, so we pass per-layer addresses.
[Differentiable]
float3 mlp_forward(Address w0, Address b0, Address w1, Address b1, Address w2, Address b2, InputVec input)
{
    let h0 = Layer0(LeakyReLU<float>(LEAKY_RELU_SLOPE)).eval<Address>(input, w0, b0);
    let h1 = Layer1(LeakyReLU<float>(LEAKY_RELU_SLOPE)).eval<Address>(h0, w1, b1);
    return outputToFloat3(Layer2(ExpActivation<float>()).eval<Address>(h1, w2, b2));
}

void getLayerAddresses(Address base, out Address w0, out Address b0,
    out Address w1, out Address b1, out Address w2, out Address b2)
{
    int offset = 0;
    w0 = base.getOffset(offset); b0 = base.getOffset(offset + INPUT_DIM * HIDDEN_DIM);
    offset = Layer0.nextOffset(offset);
    w1 = base.getOffset(offset); b1 = base.getOffset(offset + HIDDEN_DIM * HIDDEN_DIM);
    offset = Layer1.nextOffset(offset);
    w2 = base.getOffset(offset); b2 = base.getOffset(offset + HIDDEN_DIM * OUTPUT_DIM);
}

struct LatentTexture<let N : int>
{
    Tensor<float, 3> texture;
    AtomicTensor<float, 3> texture_grads;

    [Differentiable]
    float getLatent(int x, int y, int idx) { return texture[y, x, idx]; }

    [BackwardDerivativeOf(getLatent)]
    void getLatentBwd(int x, int y, int idx, float grad) { texture_grads.set({y, x, idx}, grad); }

    [Differentiable]
    InputVec sample(no_diff inout float2 uv)
    {
        int2 size = int2(texture.shape[1], texture.shape[0]);
        uv *= float2(size - 1);
        int2 c0 = clamp(int2(uv), int2(0), size - 1);
        int2 c1 = min(c0 + 1, size - 1);
        float2 f = saturate(uv - float2(c0));

        float data[INPUT_DIM];
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            data[i] = lerp(lerp(getLatent(c0.x, c0.y, i), getLatent(c1.x, c0.y, i), f.x),
                           lerp(getLatent(c0.x, c1.y, i), getLatent(c1.x, c1.y, i), f.x), f.y);
        return arrayToInputVec(data);
    }
}

struct Network
{
    LatentTexture<INPUT_DIM> latent_texture;
    RWStructuredBuffer<float>.Handle params;
    RWStructuredBuffer<float>.Handle params_grad;
}

[Differentiable]
float3 gamma_correct(float3 color) { return pow(color, 1.0f / 2.2f); }

[Differentiable]
float3 loss(Address w0, Address b0, Address w1, Address b1, Address w2, Address b2,
            LatentTexture<INPUT_DIM> tex, no_diff int2 pixel, no_diff int2 resolution, no_diff float3 reference)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float3 error = gamma_correct(mlp_forward(w0, b0, w1, b1, w2, b2, tex.sample(uv))) - gamma_correct(reference);
    return error * error;
}

struct LCG
{
    uint state;
    __init(uint seed) { state = seed; }
    [mutating] uint next_uint() { state = 1664525u * state + 1013904223u; return state; }
    [mutating] float next_float() { return (next_uint() >> 8) * 0x1p-24; }
}

float3 render(int2 pixel, int2 resolution, Network network)
{
    Address w0, b0, w1, b1, w2, b2;
    getLayerAddresses(Address(network.params), w0, b0, w1, b1, w2, b2);
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return mlp_forward(w0, b0, w1, b1, w2, b2, network.latent_texture.sample(uv));
}

// Wrapper for visualization - calls loss() without gradients
float3 show_loss(int2 pixel, int2 resolution, float3 reference, Network network)
{
    Address w0, b0, w1, b1, w2, b2;
    getLayerAddresses(Address(network.params), w0, b0, w1, b1, w2, b2);
    return loss(w0, b0, w1, b1, w2, b2, network.latent_texture, pixel, resolution, reference);
}

void calculate_grads(uint seed, int2 batch_index, int2 batch_size, Tensor<float3, 2> reference, Network network)
{
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    int2 resolution = int2(reference.shape[1], reference.shape[0]);
    int2 pixel = int2(uv * float2(resolution));
    float3 ref_color = reference.getv(pixel);

    // Build per-layer DifferentialPtrPairs (primal=params, differential=grads).
    Address pw0, pb0, pw1, pb1, pw2, pb2;
    getLayerAddresses(Address(network.params), pw0, pb0, pw1, pb1, pw2, pb2);
    Address gw0, gb0, gw1, gb1, gw2, gb2;
    getLayerAddresses(Address(network.params_grad), gw0, gb0, gw1, gb1, gw2, gb2);

    bwd_diff(loss)(
        DifferentialPtrPair<Address>(pw0, gw0), DifferentialPtrPair<Address>(pb0, gb0),
        DifferentialPtrPair<Address>(pw1, gw1), DifferentialPtrPair<Address>(pb1, gb1),
        DifferentialPtrPair<Address>(pw2, gw2), DifferentialPtrPair<Address>(pb2, gb2),
        network.latent_texture, pixel, resolution, ref_color, float3(1.0f));
}

void optimizer_step(inout float primal, inout float grad, inout float mean, inout float variance, float lr, int iter)
{
    const float BETA1 = 0.9f, BETA2 = 0.999f, EPSILON = 1e-8f;
    mean = lerp(grad, mean, BETA1);
    variance = lerp(grad * grad, variance, BETA2);
    float m_hat = mean / (1.0f - pow(BETA1, iter));
    float v_hat = variance / (1.0f - pow(BETA2, iter));
    primal -= lr * m_hat / (sqrt(v_hat) + EPSILON);
    grad = 0.0f;
}
