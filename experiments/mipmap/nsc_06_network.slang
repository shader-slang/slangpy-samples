// clang-format off
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;
import brdf;

#define PI 3.14159265358979323846f

struct MaterialParameters
{
    Tensor<float3, 2> albedo;
    Tensor<float3, 2> normal;
    Tensor<float, 2> roughness;

    float3 get_albedo(int2 pixel)
    {
        return albedo.getv(pixel);
    }
    float3 get_normal(int2 pixel)
    {
        return normalize(normal.getv(pixel));
    }
    float get_roughness(int2 pixel)
    {
        return roughness.getv(pixel);
    }
};

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

struct TwoLayerNetwork {
    NetworkParameters<2, 16> layer0;
    NetworkParameters<16, 7> layer1;

    [Differentiable]
    float[7] eval(float2 uv)
    {
        float inputs[2] = {uv.x, uv.y};
        float output0[16] = layer0.forward(inputs);
        [ForceUnroll]
        for (int i = 0; i < 16; ++i)
            output0[i] = activation(output0[i]);
        float output1[7] = layer1.forward(output0);
        for (int i = 0; i < 7; ++i)
            output1[i] = activation(output1[i]);

        return output1;
    }
}

struct ThreeLayerNetwork {
    NetworkParameters<2, 16> layer0;
    NetworkParameters<16, 16> layer1;
    NetworkParameters<16, 7> layer2;

    [Differentiable]
    float[7] eval(no_diff float2 uv)
    {
        uv = uv * 2.0f - 1.0f;
        float inputs[2] = {uv.x, uv.y};
        float output0[16] = layer0.forward(inputs);
        [ForceUnroll]
        for (int i = 0; i < 16; ++i)
            output0[i] = activation(output0[i]);
        float output1[16] = layer1.forward(output0);
        [ForceUnroll]
        for (int i = 0; i < 16; ++i)
            output1[i] = activation(output1[i]);
        float output2[7] = layer2.forward(output1);
        [ForceUnroll]
        for (int i = 0; i < 7; ++i)
            output2[i] = (sigmoid(output2[i]) - 0.5f) * 1.1f + 0.5f;

        return output2;
    }
}
struct ThreeLayerNetworkWithInputEncoding {
    NetworkParameters<16, 16> layer0;
    NetworkParameters<16, 16> layer1;
    NetworkParameters<16, 7> layer2;

    [Differentiable]
    float[7] eval(no_diff float2 uv)
    {
        //uv = uv * 2.0f;// - 1.0f;
        float inputs[16];
        [ForceUnroll]
        for (int i = 0; i < 4; ++i) {
            float scale = float(2 << i);
            inputs[i * 4 + 0] = sin(uv.x * PI * scale);
            inputs[i * 4 + 1] = cos(uv.x * PI * scale);
            inputs[i * 4 + 2] = sin(uv.y * PI * scale);
            inputs[i * 4 + 3] = cos(uv.y * PI * scale);
        }

        var output0 = layer0.forward(inputs);
        [ForceUnroll]
        for (int i = 0; i < output0.getCount(); ++i)
            output0[i] = activation(output0[i]);
        var output1 = layer1.forward(output0);
        [ForceUnroll]
        for (int i = 0; i < output1.getCount(); ++i)
            output1[i] = activation(output1[i]);
        var output2 = layer2.forward(output1);
        [ForceUnroll]
        for (int i = 0; i < 7; ++i)
            output2[i] = (sigmoid(output2[i]) - 0.5f) * 1.25f + 0.5f;

        return output2;
    }
}

typealias Network = ThreeLayerNetworkWithInputEncoding;

// Render full res BRDF from given inputs.
float3 render(int2 pixel, MaterialParameters material, float3 light_dir, float3 view_dir)
{
    return material.get_albedo(pixel);

    float light_intensity = 3;

    // Sample very shiny BRDF (it rained today!)
    float3 brdf_sample = sample_brdf(
               material.get_albedo(pixel),   // albedo color
               normalize(light_dir),         // light direction
               normalize(view_dir),          // view direction
               material.get_normal(pixel),   // normal map sample
               material.get_roughness(pixel),// roughness
               0.0,                          // metallic (no metal)
               1.0                           // specular
               );

    return brdf_sample * light_intensity;
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
float3 render_1layer(int2 pixel, int2 resolution, NetworkParameters<2, 7> network, float3 light_dir, float3 view_dir)
{
    float light_intensity = 3;

    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float[7] mat_params = network.forward({uv.x,uv.y});

    float3 albedo = float3(mat_params[0], mat_params[1], mat_params[2]);
    float3 normal = normalize(float3(mat_params[3], mat_params[4], mat_params[5]));
    float roughness = mat_params[6];

    return albedo;

    // Sample very shiny BRDF (it rained today!)
    float3 brdf_sample = sample_brdf(
               albedo,                  // albedo color
               normalize(light_dir),    // light direction
               normalize(view_dir),     // view direction
               normal,                  // normal map sample
               roughness,               // roughness
               0.0,                     // metallic (no metal)
               1.0                      // specular
               );

    return brdf_sample * light_intensity;
}

[Differentiable]
float3 loss_1layer(int2 pixel, int2 resolution, no_diff float3 reference, NetworkParameters<2, 7> network, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float3 color = render_1layer(pixel, resolution, network,
                          light_dir, view_dir);
    float3 error = color - reference;
    return error * error; // Squared error
}


// Render full res BRDF from given inputs.
[Differentiable]
float3 render_2layer(int2 pixel, int2 resolution, TwoLayerNetwork network, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float light_intensity = 3;

    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float[7] mat_params = network.eval(uv);

    float3 albedo = float3(mat_params[0], mat_params[1], mat_params[2]);
    float3 normal = normalize(float3(mat_params[3], mat_params[4], mat_params[5]));
    float roughness = mat_params[6];

    return albedo;

    // Sample very shiny BRDF (it rained today!)
    float3 brdf_sample = sample_brdf(
               albedo,                  // albedo color
               normalize(light_dir),    // light direction
               normalize(view_dir),     // view direction
               normal,                  // normal map sample
               roughness,               // roughness
               0.0,                     // metallic (no metal)
               1.0                      // specular
               );

    return brdf_sample * light_intensity;
}

[Differentiable]
float3 loss_2layer(int2 pixel, int2 resolution, no_diff float3 reference, TwoLayerNetwork network, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float3 color = render_2layer(pixel, resolution, network,
                          light_dir, view_dir);
    float3 error = color - reference;
    return error * error; // Squared error
}


// Render full res BRDF from given inputs.
[Differentiable]
float3 render_neural(int2 pixel, int2 resolution, Network network, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float light_intensity = 3;

    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float[7] mat_params = network.eval(uv);

    float3 albedo = float3(mat_params[0], mat_params[1], mat_params[2]);
    float3 normal = normalize(float3(mat_params[3], mat_params[4], mat_params[5]));
    float roughness = mat_params[6];

    return albedo;

    // Sample very shiny BRDF (it rained today!)
    float3 brdf_sample = sample_brdf(
               albedo,                  // albedo color
               normalize(light_dir),    // light direction
               normalize(view_dir),     // view direction
               normal,                  // normal map sample
               roughness,               // roughness
               0.0,                     // metallic (no metal)
               1.0                      // specular
               );

    return brdf_sample * light_intensity;
}

[Differentiable]
float3 loss_neural(int2 pixel, int2 resolution, no_diff float3 reference, Network network, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float3 color = render_neural(pixel, resolution, network,
                          light_dir, view_dir);
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

    [mutating] float3 next_dir_on_hemisphere()
    {
        // Random azimuthal angle [0, 2π)
        float phi = 2.0f * PI * next_float();

        // Random polar angle [0, π/2) for hemisphere
        float cos_theta = next_float();
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        // Convert to Cartesian coordinates
        return float3(sin_theta * cos(phi),sin_theta * sin(phi),cos_theta);
    }
};

void calculate_grads_1layer(uint seed, int2 pixel, int2 resolution, NetworkParameters<2, 7> network, MaterialParameters ref_material)
{
    LCG lcg = LCG(seed);

    for(int i = 0; i < 100; i++)
    {
        float3 light_dir = lcg.next_dir_on_hemisphere();
        float3 view_dir = lcg.next_dir_on_hemisphere();

        int2 hi_res_pixel = pixel * 4;

        float3 sum = 0;
        for(int x0 = 0; x0 < 4; x0++)
        {
            for(int y0 = 0; y0 < 4; y0++)
            {
                int2 hi_res_pixel_offset = hi_res_pixel + int2(x0, y0);
                sum += render(hi_res_pixel_offset, ref_material, light_dir, view_dir);
            }
        }
        sum /= 16.0f;

        bwd_diff(loss_1layer)(pixel, resolution, sum, network, light_dir, view_dir, 1);
    }
}

void calculate_grads_2layer(uint seed, int2 pixel, int2 resolution, TwoLayerNetwork network, MaterialParameters ref_material)
{
    LCG lcg = LCG(seed);

    pixel.x = lcg.next_uint() % resolution.x;
    pixel.y = lcg.next_uint() % resolution.y;

    float3 light_dir = lcg.next_dir_on_hemisphere();
    float3 view_dir = lcg.next_dir_on_hemisphere();

    float3 reference = render(pixel, ref_material, light_dir, view_dir);

    bwd_diff(loss_2layer)(pixel, resolution, reference, network, light_dir, view_dir, 1);
}

void calculate_grads_neural(uint seed, int2 pixel, int2 resolution, Network network, MaterialParameters ref_material)
{
    LCG lcg = LCG(seed);

    pixel.x = lcg.next_uint() % resolution.x;
    pixel.y = lcg.next_uint() % resolution.y;

    float3 light_dir = lcg.next_dir_on_hemisphere();
    float3 view_dir = lcg.next_dir_on_hemisphere();

    float3 reference = render(pixel, ref_material, light_dir, view_dir);

    bwd_diff(loss_neural)(pixel, resolution, reference, network, light_dir, view_dir, float3(1.0f));
}

// Render the difference between the rendered pixel a reference pixel.
float3 loss(int2 pixel, float3 reference, MaterialParameters material, float3 light_dir, float3 view_dir)
{
    float3 color = render(pixel, material,
                          light_dir, view_dir);
    float3 error = color - reference;
    return error * error; // Squared error
}

float3 downsample3(int2 pixel, Tensor<float3, 2> source)
{
    float3 res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}
float downsample1(int2 pixel, Tensor<float, 2> source)
{
    float res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}

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
