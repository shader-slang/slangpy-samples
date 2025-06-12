// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;
import slangpy;

__include IOptimizer;

public struct AdamOptimizer<T : IReal> : IOptimizer<T>
{
    // Internal state storing per-parameter moments
    public struct AdamState<S : IReal> : IOptimizerState<S>
    {
        S mean;
        S variance;

        public __init(S param)
        {
            mean = S(0.0f);
            variance = S(0.0f);
        }
    }
    public typealias State = AdamState<T>;

    // Batch of adam optimizer parameters, gradients and states
    public struct AdamBatch<S : IReal>
    {
        RWStructuredBuffer<S> params;
        RWStructuredBuffer<S> grads;
        RWStructuredBuffer<AdamState<S>> states;
    }
    public typealias Batch = AdamBatch<T>;

    // Adam parameters
    public T beta1;
    public T beta2;
    public T epsilon;
    public T learningRate;
    public T meanCorrectionFactor;
    public T varianceCorrectionFactor;

    public void step(inout State state, inout T param, inout T grad)
    {
        // Implement moving average via lerp operator
        state.mean = lerp(grad, state.mean, beta1);
        state.variance = lerp(grad * grad, state.variance, beta2);

        // Correct bias in mean/variance with correction factors computed on the host
        T meanCorrected = state.mean * meanCorrectionFactor;
        T varianceCorrected = state.variance * varianceCorrectionFactor;

        param -= learningRate * meanCorrected / (sqrt(varianceCorrected) + epsilon);
        grad = T(0.0f);
    }

    public void batch_step<let N : int>(Batch[N] batches, int2 batch_index)
    {
        AdamBatch<T> batch = batches[batch_index.x];

        int i = batch_index.y;

        T param = batch.params[i];
        T grad = batch.grads[i];
        AdamState<T> state = batch.states[i];

        step(state, param, grad);

        batch.params[i] = param;
        batch.grads[i] = grad;
        batch.states[i] = state;
    }
}
