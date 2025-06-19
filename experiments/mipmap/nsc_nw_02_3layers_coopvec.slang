import slangpy;
import DiffCoopVec;

#define PI 3.14159265358979323846f

struct NetworkParameters<int Inputs, int Outputs>
{
    static const CoopVecComponentType ComponentType = CoopVecComponentType.Float16;

    StructuredBuffer<half> weights, biases;
    RWStructuredBuffer<half> weightGrads, biasGrads;

    [BackwardDerivative(backward)]
    DiffCoopVec<half, Outputs> forward(DiffCoopVec<half, Inputs> x)
    {
        return DiffCoopVec<half, Outputs>(coopVecMatMulAdd<half, Outputs>(
            x.cv, ComponentType,
            weights, 0, ComponentType,
            biases, 0, ComponentType,
            CoopVecMatrixLayout.TrainingOptimal, false, 0
        ));
    }

    void backward(inout DifferentialPair<DiffCoopVec<half, Inputs>> x, DiffCoopVec<half, Outputs> grad)
    {
        coopVecOuterProductAccumulate(grad.cv, x.p.cv, weightGrads, 0, 0, CoopVecMatrixLayout.TrainingOptimal, ComponentType);
        coopVecReduceSumAccumulate(grad.cv, biasGrads, 0);

        let dX = coopVecMatMul<half, Inputs>(grad.cv, ComponentType, weights, 0, ComponentType, CoopVecMatrixLayout.TrainingOptimal, true, 0);

        x = diffPair(x.p, DiffCoopVec<half, Inputs>(dX));
    }
}


struct Network {
    NetworkParameters<16, 32>  layer0;
    NetworkParameters<32, 32> layer1;
    NetworkParameters<32, 3>  layer2;

    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        DiffCoopVec<half, 16> inputs;
        [ForceUnroll]
        for (int i = 0; i < 4; ++i) {
            float scale = float(2 << i);
            inputs[i * 4 + 0] = half(sin(uv.x * PI * scale));
            inputs[i * 4 + 1] = half(cos(uv.x * PI * scale));
            inputs[i * 4 + 2] = half(sin(uv.y * PI * scale));
            inputs[i * 4 + 3] = half(cos(uv.y * PI * scale));
        }

        var output0 = layer0.forward(inputs);
        output0 = leakyReLU(output0);
        var output1 = layer1.forward(output0);
        output1 = leakyReLU(output1);
        var output2 = layer2.forward(output1);
        output2 = exp(output2);
        return float3(output2.toVector());
    }
}

[Differentiable]
DiffCoopVec<half, N> activation<int N>(DiffCoopVec<half, N> x)
{
    return max(x, DiffCoopVec<half, N>(0.0h));
}

[Differentiable]
DiffCoopVec<half, N> leakyReLU<int N>(DiffCoopVec<half, N> x)
{
    return max(x, DiffCoopVec<half, N>(0.0h)) + min(x, DiffCoopVec<half, N>(0.0h)) * 0.01h;
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

void optimize1(inout half primalH, inout half gradH, inout float m_prev, inout float v_prev, float learning_rate, int iteration)
{
    // Standard Adam default values.
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    float primal = float(primalH);
    float grad = float(gradH);

    if (isnan(grad) || isinf(grad))
        grad = 0.0h;

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

    primalH = half(primal);
    gradH = 0.0h;
}

void calculate_grads(uint seed, int2 pixel, int2 resolution, float3 reference, Network network)
{
    bwd_diff(loss)(pixel, resolution, reference, network, 1.0f);
}
