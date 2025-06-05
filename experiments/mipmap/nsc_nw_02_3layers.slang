import slangpy;

struct NetworkParameters<int Inputs, int Outputs>
{
    RWTensor<float, 1> biases;
    RWTensor<float, 2> weights;

    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable]
    float get_bias(int neuron)
    {
        return biases.get({neuron});
    }
    [Differentiable]
    float get_weight(int neuron, int input)
    {
        return weights.get({neuron, input});
    }

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float grad)
    {
        biases_grad.set({neuron}, grad);
    }

    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float grad)
    {
        weights_grad.set({neuron, input}, grad);
    }

    // ... Fields and accessors here ...

    [Differentiable]
    float[Outputs] forward(float[Inputs] x)
    {
        float[Outputs] y;
        [ForceUnroll]
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
}

#define PI 3.14159265358979323846f

struct Network {
    RWTensor<float, 2> latent;
    AtomicTensor<float, 2> latent_grad;

    NetworkParameters<17, 16>  layer0;
    NetworkParameters<16, 16> layer1;
    NetworkParameters<16, 3>  layer2;

    [Differentiable]
    float get_latent(int2 index)
    {
        return latent.getv(index);
    }
    [BackwardDerivativeOf(get_latent)]
    void get_latent_bwd(int2 index, float grad)
    {
        latent_grad.setv(index, grad);
    }

    [Differentiable]
    float sample_latent(float2 uv)
    {
        int2 shape = int2(latent.shape[1], latent.shape[0]);
        int2 maxidx = shape - 1;
        int2 texel = int2(uv * shape);
        float2 frac = frac(uv * shape);

        float c00 = get_latent(texel);
        float c10 = get_latent(min(texel + int2(1, 0),maxidx));
        float c01 = get_latent(min(texel + int2(0, 1),maxidx));
        float c11 = get_latent(min(texel + int2(1, 1),maxidx));

        return lerp(lerp(c00, c10, frac.x), lerp(c01, c11, frac.x), frac.y);
    }

    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        float latent_value = sample_latent(uv);

        float inputs[17];
        [ForceUnroll]
        for (int i = 0; i < 4; ++i) {
            float scale = float(2 << i);
            inputs[i * 4 + 0] = sin(uv.x * PI * scale);
            inputs[i * 4 + 1] = cos(uv.x * PI * scale);
            inputs[i * 4 + 2] = sin(uv.y * PI * scale);
            inputs[i * 4 + 3] = cos(uv.y * PI * scale);
        }
        inputs[16] = latent_value;

        float output0[16] = layer0.forward(inputs);
        [ForceUnroll]
        for (int i = 0; i < 16; ++i)
            output0[i] = activation(output0[i]);
        float output1[16] = layer1.forward(output0);
        [ForceUnroll]
        for (int i = 0; i < 16; ++i)
            output1[i] = activation(output1[i]);
        float output2[3] = layer2.forward(output1);
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            output2[i] = sigmoid(output2[i]);
        return float3(output2[0], output2[1], output2[2]);
    }
}

[Differentiable]
float activation(float x)
{
    return max(x, 0.0f);
}

[Differentiable]
float sigmoid(float x)
{
    float expX = exp(x);
    return expX / (expX + 1.0f);
}

// Render full res BRDF from given inputs.
[Differentiable]
float3 render(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}

[Differentiable]
float3 loss(int2 pixel, int2 resolution, no_diff float3 reference, Network network)
{
    float3 color = render(pixel, resolution, network);
    float3 error = color - reference;
    return error * error; // Squared error
}

struct LCG
{
    uint state;

    __init(uint seed) { state = seed; }

    [mutating]
    uint next_uint()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_float()
    {
        // Convert to float in range [0, 1)
        return (next_uint() >> 8) * 0x1p-24;
    }
};

void optimize1(inout float primal, inout float grad, inout float m_prev, inout float v_prev, float learning_rate, int iteration)
{
    // Standard Adam default values.
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    if (isnan(grad) || isinf(grad))
        grad = 0.0f;

    // Adam optimization.
    float gradient2 = grad * grad;

    float m = ADAM_BETA_1 * m_prev + (1.0 - ADAM_BETA_1) * grad;
    float v = ADAM_BETA_2 * v_prev + (1.0 - ADAM_BETA_2) * gradient2;

    m_prev = m;
    v_prev = v;

    float mHat = m / (1.0f - pow(ADAM_BETA_1, iteration));
    float vHat = v / (1.0f - pow(ADAM_BETA_2, iteration));

    float update = learning_rate * (mHat / (sqrt(vHat) + ADAM_EPSILON));

    // Subtract the optimized result from the trained normal and reset the gradient.
    primal -= update;

    grad = 0;
}

void calculate_grads(uint seed, int2 pixel, int2 resolution, float3 reference, Network network)
{
    bwd_diff(loss)(pixel, resolution, reference, network, 1.0f);
}
