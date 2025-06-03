// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;
import brdf;

struct MaterialParameters
{
    RWTensor<float3, 2> albedo;
    RWTensor<float3, 2> normal;

    RWTensor<float3, 2> albedo_grad;
    RWTensor<float3, 2> normal_grad;

    [Differentiable]
    float3 get_albedo(int2 pixel)
    {
        return albedo.getv(pixel);
    }

    [BackwardDerivativeOf(get_albedo)]
    void get_albedo_bwd(int2 pixel, float3 grad)
    {
        albedo_grad.setv(pixel, albedo_grad.getv(pixel)+grad);
    }

    [Differentiable]
    float3 get_normal(int2 pixel)
    {
        return normal.getv(pixel);
    }

    [BackwardDerivativeOf(get_normal)]
    void get_normal_bwd(int2 pixel, float3 grad)
    {
        normal_grad.setv(pixel, normal_grad.getv(pixel)+grad);
    }
};

// Render full res BRDF from given inputs.
[Differentiable]
float3 render(int2 pixel, no_diff MaterialParameters material, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float light_intensity = 3;

    // Sample very shiny BRDF (it rained today!)
    float3 brdf_sample = sample_brdf(
               material.get_albedo(pixel),  // albedo color
               normalize(light_dir),        // light direction
               normalize(view_dir),         // view direction
               normalize(material.get_normal(pixel)),  // normal map sample
               0.3,                         // roughness
               0.0,                         // metallic (no metal)
               1.0                          // specular
               );

    return brdf_sample * light_intensity;
}

// Render the difference between the rendered pixel a reference pixel.
[Differentiable]
float3 loss(int2 pixel, no_diff float3 reference, no_diff MaterialParameters material, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float3 color = render(pixel, material,
                          light_dir, view_dir);
    float3 error = color - reference;
    error = abs(error);
    return (error.x + error.y + error.z) / 3.0f; // Average RGB error
}

#define PI 3.14159265358979323846f

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

void calculate_grads(uint seed, int2 pixel, MaterialParameters material, MaterialParameters ref_material)
{
    LCG lcg = LCG(seed);

    for(int i = 0; i < 10000; i++)
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


        bwd_diff(loss)(pixel, sum, material, light_dir, view_dir, 1);
    }
    material.normal_grad.setv(pixel,material.normal_grad.getv(pixel)/10000.0f);
}

void init_normal(inout float3 val)
{
    val = float3(0,0,1);
}

void scale(inout float3 val, float k)
{
    val = val * k;
}


void optimize(inout float3 primal, inout float3 grad, inout float3 m_prev, inout float3 v_prev, float learning_rate, bool renormalize)
{
    // Standard Adam default values.
    const float ADAM_ETA = 0.002;
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    // Adam optimization.
    float3 gradient2 = grad * grad;

    float3 m = ADAM_BETA_1 * m_prev + (1.0 - ADAM_BETA_1) * grad;
    float3 v = ADAM_BETA_2 * v_prev + (1.0 - ADAM_BETA_2) * gradient2;

    m_prev = m;
    v_prev = v;

    float3 mHat = m / (1.0f - ADAM_BETA_1);
    float3 vHat = v / (1.0f - ADAM_BETA_2);

    float3 update = (ADAM_ETA / (sqrt(vHat) + ADAM_EPSILON)) * mHat;

    // Subtract the optimized result from the trained normal and reset the gradient.
    primal -= update;
    if(renormalize)
        primal = normalize(primal); // Ensure the normal is still normalized.
    //primal = primal; // Ensure the normal is still normalized.

    grad = float3(0.0f);
}

float3 downsample(int2 pixel, Tensor<float3, 2> source)
{
    float3 res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}
