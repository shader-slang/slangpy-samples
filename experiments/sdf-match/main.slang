// SPDX-License-Identifier: Apache-2.0
import rayMarching;
import tinymlp;
import slangpy;

public struct AdamState
{
    public float mean;
    public float variance;
    public float epsilon;
    public int iteration;
    public float learning_rate;
}

public struct TinyMLP_Params
{
    public float* m_params;
    public float* m_grads;
}

public struct AdamOptimizer
{
    public static void update(inout float param, inout float grad, inout AdamState state, int batch_size, float beta1, float beta2)
    {
        grad = grad / batch_size;

        float grad_clip = 5.0f;
        grad = clamp(grad, -grad_clip, grad_clip);

        state.mean = beta1 * state.mean + (1.0f - beta1) * grad;
        state.variance = beta2 * state.variance + (1.0f - beta2) * grad * grad;
        state.iteration++;

        float m_hat = state.mean / (1.0f - pow(beta1, state.iteration));
        float v_hat = state.variance / (1.0f - pow(beta2, state.iteration));
        float denom = sqrt(v_hat) + state.epsilon;

        param -= state.learning_rate * m_hat / denom;
        grad = 0.0f;
    }
}

public void clearAdamState(inout AdamState state)
{
    state.mean = 0.0f;
    state.variance = 0.0f;
    state.iteration = 0;
    state.learning_rate = 2e-3f;
    state.epsilon = 1e-8f;
}

public void updateParams(TinyMLP_Params* params, inout AdamState state, int batch_size, float beta1, float beta2)
{
    int call_id = CallShapeInfo::get_call_id().shape[0];
    AdamOptimizer::update(params->m_params[call_id], params->m_grads[call_id], state, batch_size, beta1, beta2);
}



// target function is just a simplified version of targetSDF, we don't want to use targetSDF
// directly because it has a square root operation, which is not stable during training
public float[output_size] targetFunction(float[input_size] input)
{
    float[output_size] output;
    [ForceUnroll]
    for (int i = 0; i < output_size; i++)
    {
        float sum = 0.0f;
        [ForceUnroll]
        for (int j = 0; j < input_size; j++)
        {
            sum += input[j] * input[j];
        }
        output[i] = sum;
    }
    return output;
}

[Differentiable]
internal float loss(Ref<TinyMLP> mlp, float[input_size] input)
{
    float[output_size] output = mlp.eval(input);
    float[output_size] target = no_diff targetFunction(input);
    float[output_size] diff;

    float sum = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < output_size; i++)
    {
        diff[i] = output[i] - target[i];
        sum += diff[i] * diff[i];
    }

    return sum / output_size;
}

public float trainMLP(
    TinyMLP_Params* params,
    float[input_size] input)
{
    TinyMLP mlp = TinyMLP(params);
    var d_input = diffPair(input);
    bwd_diff(loss)(mlp, d_input, 1.0f);

    float current_loss = loss(mlp, input);
    return current_loss;
}

public float4 RunRayMarch(float2 screenSize, int2 gridId, TinyMLP_Params* params)
{
    if (gridId.x >= screenSize.x || gridId.y >= screenSize.y)
        return float4(0.0);

    float2 uv = (gridId.xy * 1.0f / screenSize.xy) * 2.0 - 1.0;    // Normalized screen coordinates to [-1, 1]
    uv.x *= screenSize.x / screenSize.y; // Maintain aspect rati

    float3 rayOrigin = cameraPosition;
    float3 rayDir = normalize(float3(uv, 1.0));
    TinyMLP mlp = TinyMLP(params);

    ISDFModel sdfModel[2] = {
        TargetSDFModle(float3(0.5, 0.6, 1.0)),
        LearnedSDFModle(mlp, float3(1.0, 0.6, 0.3))
    };

    Ray ray = {rayOrigin, rayDir};
    RayMarcher rayMarcher = {ray};

    float3 color = float3(0.0f);
    for (uint i = 0; i < sdfModel.getCount(); i++)
    {
        color += rayMarcher.getModelColor(sdfModel[i]);
    }

    color = color / sdfModel.getCount();
    return float4(color, 1.0);
}
