// SPDX-License-Identifier: Apache-2.0
import slangpy;
import neural;

static const int INPUT_DIM = 4;
static const int HIDDEN_DIM = 32;
static const int OUTPUT_DIM = 3;
static const float LEAKY_RELU_SLOPE = 0.01f;

typealias InputVec = InlineVector<float, INPUT_DIM>;
typealias HiddenVec = InlineVector<float, HIDDEN_DIM>;
typealias OutputVec = InlineVector<float, OUTPUT_DIM>;

typealias Storage = StructuredBufferStorage<float>;
typealias Layer0 = FFLayer<float, InputVec, HiddenVec, Storage, LeakyReLU<float>, true>;
typealias Layer1 = FFLayer<float, HiddenVec, HiddenVec, Storage, LeakyReLU<float>, true>;
typealias Layer2 = FFLayer<float, HiddenVec, OutputVec, Storage, ExpActivation<float>, true>;

static const uint TOTAL_PARAMS = Layer0.ParameterCount + Layer1.ParameterCount + Layer2.ParameterCount;

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

struct LatentTexture<let N : int>
{
    Tensor<float, 3> texture;
    AtomicTensor<float, 3> texture_grads;

    [Differentiable]
    float getLatent(int x, int y, int idx) { return texture[y, x, idx]; }

    [BackwardDerivativeOf(getLatent)]
    void getLatentBwd(int x, int y, int idx, float grad) { texture_grads.set({y, x, idx}, grad); }

    [Differentiable]
    InlineVector<float, N> sample(no_diff inout float2 uv)
    {
        int2 size = int2(texture.shape[1], texture.shape[0]);
        uv *= float2(size - 1);
        int2 c0 = clamp(int2(uv), int2(0), size - 1);
        int2 c1 = min(c0 + 1, size - 1);
        float2 f = saturate(uv - float2(c0));

        InlineVector<float, N> result;
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            result.set(i, lerp(lerp(getLatent(c0.x, c0.y, i), getLatent(c1.x, c0.y, i), f.x),
                               lerp(getLatent(c0.x, c1.y, i), getLatent(c1.x, c1.y, i), f.x), f.y));
        return result;
    }
}

struct Network
{
    LatentTexture<INPUT_DIM> latent_texture;
    RWStructuredBuffer<float> params;
    RWStructuredBuffer<float> params_grad;
}

[Differentiable]
float3 eval(Storage storage, LatentTexture<INPUT_DIM> tex, no_diff float2 uv)
{
    let out = mlp_forward(storage, tex.sample(uv));
    return float3(out.get(0), out.get(1), out.get(2));
}

[Differentiable]
float3 gamma_correct(float3 color) { return pow(color, 1.0f / 2.2f); }

[Differentiable]
float3 loss(Storage storage, LatentTexture<INPUT_DIM> tex, no_diff int2 pixel, no_diff int2 resolution, no_diff float3 reference)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float3 error = gamma_correct(eval(storage, tex, uv)) - gamma_correct(reference);
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
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return eval(Storage(network.params), network.latent_texture, uv);
}

// Wrapper for visualization - calls loss() without gradients
float3 show_loss(int2 pixel, int2 resolution, float3 reference, Network network)
{
    return loss(Storage(network.params), network.latent_texture, pixel, resolution, reference);
}

void calculate_grads(uint seed, int2 batch_index, int2 batch_size, Tensor<float3, 2> reference, Network network)
{
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    int2 resolution = int2(reference.shape[1], reference.shape[0]);
    int2 pixel = int2(uv * float2(resolution));
    float3 ref_color = reference.getv(pixel);

    let storage_pair = DifferentialPtrPair<Storage>(Storage(network.params), Storage(network.params_grad));
    bwd_diff(loss)(storage_pair, network.latent_texture, pixel, resolution, ref_color, float3(1.0f));
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
