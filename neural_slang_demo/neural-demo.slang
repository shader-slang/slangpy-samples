// SPDX-License-Identifier: Apache-2.0
import slangpy;
import neural;

// =============================================================================
// Neural.slang Types
// =============================================================================
typealias Vec4 = InlineVector<float, 4>;
typealias Vec32 = InlineVector<float, 32>;
typealias Vec3 = InlineVector<float, 3>;

typealias Storage = StructuredBufferStorage<float>;

// FFLayer types with activations built-in
typealias Layer0Type = FFLayer<float, Vec4, Vec32, Storage, LeakyReLU<float>, true>;
typealias Layer1Type = FFLayer<float, Vec32, Vec32, Storage, LeakyReLU<float>, true>;
typealias Layer2Type = FFLayer<float, Vec32, Vec3, Storage, ExpActivation<float>, true>;

static const int INPUT_SIZE = Vec4.Size;
static const int HIDDEN_SIZE = Vec32.Size;
static const int OUTPUT_SIZE = Vec3.Size;
static const float LEAKY_ALPHA = 0.01f;

// Total parameter count for single buffer
static const uint TOTAL_PARAMS = Layer0Type.ParameterCount + Layer1Type.ParameterCount + Layer2Type.ParameterCount;

// =============================================================================
// MLP Network using FFLayer with single buffer
// =============================================================================
struct MLPNetwork
{
    RWStructuredBuffer<float> params;  // Single buffer for all layers

    Vec3 forward(Vec4 input)
    {
        let storage = Storage(params);

        // Chain layers using nextAddress
        let addr0 = 0u;
        let ff0 = Layer0Type(addr0, addr0 + INPUT_SIZE * HIDDEN_SIZE);

        let addr1 = Layer0Type.nextAddress(addr0);
        let ff1 = Layer1Type(addr1, addr1 + HIDDEN_SIZE * HIDDEN_SIZE);

        let addr2 = Layer1Type.nextAddress(addr1);
        let ff2 = Layer2Type(addr2, addr2 + HIDDEN_SIZE * OUTPUT_SIZE);

        // Activations are now built into FFLayer (Option 2: storage as parameter)
        Vec32 h0 = ff0.eval<Storage>(storage, LEAKY_ALPHA, input);
        Vec32 h1 = ff1.eval<Storage>(storage, LEAKY_ALPHA, h0);
        Vec3 out = ff2.eval<Storage>(storage, NoParam(), h1);

        return out;
    }
}

// =============================================================================
// Latent Texture
// - TODO: This could potentially be replaced once we have the hash grid implementation inside neural.slang
// =============================================================================
struct LatentTexture<int NumLatents>
{
    Tensor<float, 3> texture;
    AtomicTensor<float, 3> texture_grads;

    [Differentiable]
    float getLatent(int x, int y, int idx) { return texture[y, x, idx]; }

    [BackwardDerivativeOf(getLatent)]
    void getLatentBwd(int x, int y, int idx, float grad) { texture_grads.set({y, x, idx}, grad); }

    [Differentiable]
    Vec4 sample(no_diff inout float2 uv)
    {
        int2 size = {texture.shape[1], texture.shape[0]};
        uv *= float2(size - 1);
        int2 c0 = clamp(int2(uv), int2(0), size - 1);
        int2 c1 = min(c0 + 1, size - 1);
        uv = saturate(uv - float2(c0));

        Vec4 result;
        [ForceUnroll]
        for (int i = 0; i < NumLatents; ++i)
        {
            float v00 = getLatent(c0.x, c0.y, i);
            float v10 = getLatent(c1.x, c0.y, i);
            float v01 = getLatent(c0.x, c1.y, i);
            float v11 = getLatent(c1.x, c1.y, i);
            result[i] = lerp(lerp(v00, v10, uv.x), lerp(v01, v11, uv.x), uv.y);
        }
        return result;
    }
}

// =============================================================================
// Network
// =============================================================================
struct Network
{
    static const int NumLatents = INPUT_SIZE;
    LatentTexture<NumLatents> latent_texture;
    MLPNetwork mlp;

    float3 eval(float2 uv)
    {
        Vec4 latents = latent_texture.sample(uv);
        Vec3 output = mlp.forward(latents);
        return float3(output[0], output[1], output[2]);
    }
}

// =============================================================================
// Trainable MLP (for gradient accumulation)
// =============================================================================
// TrainableLayer using raw arrays instead of InlineVector for differentiability
struct TrainableLayer<int Inputs, int Outputs>
{
    Tensor<float, 2> weights;
    Tensor<float, 1> biases;
    AtomicTensor<float, 2> weights_grad;
    AtomicTensor<float, 1> biases_grad;

    [Differentiable]
    float getWeight(int row, int col) { return weights.get({row, col}); }

    [Differentiable]
    float getBias(int row) { return biases.get({row}); }

    [BackwardDerivativeOf(getWeight)]
    void getWeightBwd(int row, int col, float grad) { weights_grad.set({row, col}, grad); }

    [BackwardDerivativeOf(getBias)]
    void getBiasBwd(int row, float grad) { biases_grad.set({row}, grad); }

    // Forward using raw float array - fully differentiable
    [Differentiable]
    void forward_arr(float input[Inputs], out float output[Outputs])
    {
        [ForceUnroll]
        for (int row = 0; row < Outputs; ++row)
        {
            var sum = getBias(row);
            [ForceUnroll]
            for (int col = 0; col < Inputs; ++col)
                sum += getWeight(row, col) * input[col];
            output[row] = sum;
        }
    }
}

struct TrainableMLPNetwork
{
    TrainableLayer<INPUT_SIZE, HIDDEN_SIZE> layer0;
    TrainableLayer<HIDDEN_SIZE, HIDDEN_SIZE> layer1;
    TrainableLayer<HIDDEN_SIZE, OUTPUT_SIZE> layer2;

    // Helper: LeakyReLU on array
    [Differentiable]
    void leaky_relu<let N : int>(float input[N], out float output[N])
    {
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
            output[i] = input[i] > 0.0f ? input[i] : LEAKY_ALPHA * input[i];
    }

    // Compute loss directly using raw arrays - fully differentiable
    [Differentiable]
    float forward_loss(Vec4 input_vec, no_diff float3 reference)
    {
        // Convert Vec4 to array (using no_diff since we don't need gradient for input)
        float input[INPUT_SIZE];
        [ForceUnroll]
        for (int i = 0; i < INPUT_SIZE; ++i)
            input[i] = no_diff input_vec[i];
        
        // Layer 0: input -> h0
        float h0_raw[HIDDEN_SIZE];
        layer0.forward_arr(input, h0_raw);
        float h0[HIDDEN_SIZE];
        leaky_relu<HIDDEN_SIZE>(h0_raw, h0);
        
        // Layer 1: h0 -> h1
        float h1_raw[HIDDEN_SIZE];
        layer1.forward_arr(h0, h1_raw);
        float h1[HIDDEN_SIZE];
        leaky_relu<HIDDEN_SIZE>(h1_raw, h1);
        
        // Layer 2: h1 -> output (with exp activation)
        float output_raw[OUTPUT_SIZE];
        layer2.forward_arr(h1, output_raw);
        
        // Apply exp activation and compute loss
        float pred0 = exp(output_raw[0]);
        float pred1 = exp(output_raw[1]);
        float pred2 = exp(output_raw[2]);
        
        // Gamma correct and compute MSE
        float err0 = pow(pred0, 1.0f / 2.2f) - pow(reference.x, 1.0f / 2.2f);
        float err1 = pow(pred1, 1.0f / 2.2f) - pow(reference.y, 1.0f / 2.2f);
        float err2 = pow(pred2, 1.0f / 2.2f) - pow(reference.z, 1.0f / 2.2f);
        
        return err0 * err0 + err1 * err1 + err2 * err2;
    }
}

struct TrainableNetwork
{
    static const int NumLatents = INPUT_SIZE;
    LatentTexture<NumLatents> latent_texture;
    TrainableMLPNetwork mlp;

    // Compute loss directly - avoids Vec->float3 conversion issue
    [Differentiable]
    float eval_loss(no_diff float2 uv, no_diff float3 reference)
    {
        Vec4 latents = latent_texture.sample(uv);
        return mlp.forward_loss(latents, reference);
    }
}

// =============================================================================
// Rendering and Loss
// =============================================================================
float3 render(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}

[Differentiable]
float3 gamma_correct(float3 color) { return pow(color, 1.0f / 2.2f); }

float3 loss(int2 pixel, int2 resolution, float3 reference, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float3 color = network.eval(uv);
    float3 error = gamma_correct(color) - gamma_correct(reference);
    return error * error;
}

// loss_trainable returns scalar float (computed inside network to avoid Vec->float3 issue)
[Differentiable]
float loss_trainable(int2 pixel, int2 resolution, no_diff float3 reference, TrainableNetwork network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval_loss(uv, reference);
}

// =============================================================================
// Utilities
// =============================================================================
struct LCG
{
    uint state;
    __init(uint seed) { state = seed; }
    [mutating] uint next_uint() { state = 1664525u * state + 1013904223u; return state; }
    [mutating] float next_float() { return (next_uint() >> 8) * 0x1p-24; }
};

void optimizer_step(inout float primal, inout float grad, inout float mean, inout float variance, float lr, int iter)
{
    const float B1 = 0.9f, B2 = 0.999f, EPS = 1e-8f;
    mean = lerp(grad, mean, B1);
    variance = lerp(grad * grad, variance, B2);
    float mHat = mean / (1.0f - pow(B1, iter));
    float vHat = variance / (1.0f - pow(B2, iter));
    primal -= lr * mHat / (sqrt(vHat) + EPS);
    grad = 0.0f;
}

void calculate_grads(uint seed, int2 batch_index, int2 batch_size, Tensor<float3, 2> reference, TrainableNetwork network)
{
    LCG lcg = LCG(seed);
    float2 uv = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    int2 resolution = {reference.shape[1], reference.shape[0]};
    int2 pixel = int2(uv * float2(resolution));
    float3 ref_color = reference.getv(pixel);
    // Pass 1.0 as gradient for scalar loss
    bwd_diff(loss_trainable)(pixel, resolution, ref_color, network, 1.0f);
}
