// SPDX-License-Identifier: Apache-2.0

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

struct Network {
    NetworkParameters<2, 32>  layer0;
    NetworkParameters<32, 32> layer1;
    NetworkParameters<32, 3>  layer2;

    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        float inputs[2] = {uv.x, uv.y};
        float output0[32] = layer0.forward(inputs);
        [ForceUnroll]
        for (int i = 0; i < 32; ++i)
            output0[i] = activation(output0[i]);
        float output1[32] = layer1.forward(output0);
        [ForceUnroll]
        for (int i = 0; i < 32; ++i)
            output1[i] = activation(output1[i]);
        float output2[3] = layer2.forward(output1);
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            output2[i] = activation(output2[i]);
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
