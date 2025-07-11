import slangpy;

static const int input_size = 3;
static const int hidden_size = 16;
static const int output_size = 1;

public struct Layer<int input_size, int output_size>
{
    internal float* m_weights;
    internal float* m_biases;
    internal float* m_weights_grad;
    internal float* m_biases_grad;

    static const float LEAKY_RELU_ALPHA = 0.01f;

    internal __init(float* inWeights, float* inBiases, float* inWeightsGrad, float* inBiasesGrad)
    {
        m_weights = inWeights;
        m_biases = inBiases;
        m_weights_grad = inWeightsGrad;
        m_biases_grad = inBiasesGrad;
    }

    [BackwardDerivative(backward)]
    [NoDiffThis]
    public float[output_size] forward(float[input_size] input)
    {
        float[output_size] output;
        [ForceUnroll]
        for (int i = 0; i < output_size; i++) {
            float sum = 0.0f;
            [ForceUnroll]
            for (int j = 0; j < input_size; j++) {
                sum += m_weights[i * input_size + j] * input[j];
            }
            output[i] = sum + m_biases[i];

            // apply leaky relu on output
            if (output[i] < 0.0f)
            {
                output[i] *= LEAKY_RELU_ALPHA;
            }
        }
        return output;
    }

    public void backward(inout DifferentialPair<float[input_size]> input, float[output_size] dResult)
    {
        float[output_size] output = forward(input.p);

        [ForceUnroll]
        for (int i = 0; i < output_size; i++)
        {
            // apply leaky relu on output
            if (output[i] < 0.0f)
            {
                dResult[i] *= LEAKY_RELU_ALPHA;
            }
        }

        // gradient of weights += input^T * dResult
        [ForceUnroll]
        for (int i = 0; i < input_size; i++)
        {
            for (int j = 0; j < output_size; j++)
            {
                // Gradient of weights is shared by all input elements, so accumulation has to be atomic
                __atomic_add(m_weights_grad[i * output_size + j], dResult[j] * input.p[i]);
            }
        }

        // gradient of biases += dResult
        [ForceUnroll]
        for (int i = 0; i < output_size; i++)
        {
            // Gradient of biases is shared by all input elements, so accumulation has to be atomic
            __atomic_add(m_biases_grad[i], dResult[i]);
        }

        // gradient of input = weight^T * dResult
        float[input_size] dInput = {};

        [ForceUnroll]
        for (int i = 0; i < input_size; i++)
        {
            float sum = 0.0f;
            [ForceUnroll]
            for (int j = 0; j < output_size; j++)
            {
                sum += m_weights[j * input_size + i] * dResult[j];
            }
            dInput[i] += sum;
        }

        input = {input.p, dInput};
    }
}

public struct TinyMLP_Params
{
    internal float* m_params;
    internal float* m_grads;
}

public struct TinyMLP
{
    internal Layer<input_size, hidden_size> m_layer1;
    internal Layer<hidden_size, output_size> m_layer2;

    public __init(TinyMLP_Params* params)
    {
        {
            constexpr int weights_stride = input_size * hidden_size;
            constexpr int biases_offset = weights_stride;
            m_layer1 = Layer<input_size, hidden_size>(params->m_params, params->m_params + biases_offset,
            params->m_grads, params->m_grads + biases_offset);
        }

        {
            constexpr int layer_offset = (input_size * hidden_size) + hidden_size;  // offset of layer 2
            constexpr int weights_stride = hidden_size * output_size;               // stride of weights of layer 2
            constexpr int biases_offset = layer_offset + weights_stride;            // offset of biases of layer 2

            m_layer2 = Layer<hidden_size, output_size>(params->m_params + layer_offset, params->m_params + biases_offset,
            params->m_grads + layer_offset, params->m_grads + biases_offset);
        }
    }

    [Differentiable]
    [NoDiffThis]
    public float[output_size] eval(float[input_size] input)
    {
        float[hidden_size] hidden = m_layer1.forward(input);
        return m_layer2.forward(hidden);
    }
}

public struct AdamState
{
    internal float mean;
    internal float variance;
    internal float epsilon;
    internal int iteration;
    internal float learning_rate;
}

public struct AdamOptimizer
{
    public static void update(inout float param, inout float grad, inout AdamState state, float beta1, float beta2)
    {
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

float[output_size] targetFunc(float[input_size] input)
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
    float[output_size] target = no_diff targetFunc(input);

    float[output_size] diff;

    float sum = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < output_size; i++)
    {
        diff[i] = output[i] - target[i];
        sum += diff[i] * diff[i];
    }

    return sqrt(sum);
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

public void clearAdamState(inout AdamState state)
{
    state.mean = 0.0f;
    state.variance = 0.0f;
    state.iteration = 0;
    state.learning_rate = 1e-3f;
    state.epsilon = 1e-8f;
}

public int updateParams(TinyMLP_Params* params, inout AdamState state, float beta1, float beta2)
{
    int call_id = CallShapeInfo::get_call_id().shape[0];
    AdamOptimizer::update(params->m_params[call_id], params->m_grads[call_id], state, beta1, beta2);

    return call_id;
}

public float[output_size] EvalMLP(TinyMLP_Params* params, float[input_size] input)
{
    TinyMLP mlp = TinyMLP(params);
    return mlp.eval(input);
}

public float returnMLP(TinyMLP_Params* params)
{
    return params->m_params[1];
}