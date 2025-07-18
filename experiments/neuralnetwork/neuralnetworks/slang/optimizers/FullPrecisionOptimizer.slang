// SPDX-License-Identifier: Apache-2.0

implementing NeuralNetworks;

__include IOptimizer;

/*
    Runs a nested optimizer (e.g. Adam) on a full-precision copy of the model's parameters,
    then converts it to half precision after each step(). This avoids accruing half precision
    rounding errors during training.

    Floating point parameters are optimized directly, half parameters are stored together with
    a floating point copy.
*/
public struct FullPrecisionOptimizer<T : IReal, NestedOptimizer : IOptimizer<float>>
{
    public float gradientScale;
    public NestedOptimizer nestedOptim;
}

// Extension: FullPrecisionOptimizer run on a full precision parameter
// => Do nothing special, just optimize the parameter directly with the nested optimizer.
public extension<NestedOptimizer : IOptimizer<float>> FullPrecisionOptimizer<float, NestedOptimizer> : IOptimizer<float>
{
    public typealias State = NestedOptimizer::State;
    public typealias Batch = NestedOptimizer::Batch;

    public void step(inout State state, inout float param, inout float grad)
    {
        nestedOptim.step(state, param, grad);
    }
    public void batch_step<let N : int>(Batch[N] batches, int2 batch_index)
    {
        nestedOptim.batch_step(batches, batch_index);
    }
}

// Extension: FullPrecisionOptimizer run on a half precision parameter
// => Store a full precision copy, optimize that instead, convert to the half precision parameter
// after each update
public extension<NestedOptimizer : IOptimizer<float>> FullPrecisionOptimizer<half, NestedOptimizer> : IOptimizer<half>
{
    public typealias State = FullPrecisionParameterHalfState<NestedOptimizer>;
    public typealias Batch = FullPrecisionParameterHalfBatch<NestedOptimizer>;

    public void step(inout State state, inout half paramH, inout half gradH)
    {
        float gradF = float(gradH) / gradientScale;
        nestedOptim.step(state.nestedState, state.paramF, gradF);
        gradH = half(gradF);
        paramH = half(state.paramF);
    }

    public void batch_step<let N : int>(Batch[N] batches, int2 batch_index)
    {
        Batch batch = batches[batch_index.x];

        int i = batch_index.y;

        half param = batch.params[i];
        half grad = batch.grads[i];
        State state = batch.states[i];

        step(state, param, grad);

        batch.params[i] = param;
        batch.grads[i] = grad;
        batch.states[i] = state;
    }
}

// This struct definition should ideally move into the extension for T==half,
// (and avoid the NestedOptimizer generic param), but this triggers two separate
// reflection bugs (slang issue #6544, #6546)
public struct FullPrecisionParameterHalfState<NestedOptimizer : IOptimizer<float>> : IOptimizerState<half>
{
    public float paramF;

    public NestedOptimizer::State nestedState;

    public __init(half paramH)
    {
        paramF = float(paramH);
        nestedState = NestedOptimizer::State(paramF);
    }
}
public struct FullPrecisionParameterHalfBatch<NestedOptimizer : IOptimizer<float>>
{
    RWStructuredBuffer<half> params;
    RWStructuredBuffer<half> grads;
    RWStructuredBuffer<FullPrecisionParameterHalfState<NestedOptimizer>> states;
}
