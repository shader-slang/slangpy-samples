// SPDX-License-Identifier: Apache-2.0

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
        int iteration;

        public __init(S param)
        {
            mean = S(0.0f);
            // variance = S(-1.0f);
            variance = S(0.0f);
            iteration = 0;
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

    public void step(inout State state, inout T param, inout T grad)
    {
        // Note: The standard Adam implementation contains bias in the mean and variance
        // This is because they are a moving average and are initialized to zero,
        // leading to a bias towards zero that diminishes over time.
        // The original Adam paper suggests to cancel out the bias by dividing out
        // the weight of the zero term, computed with a power of the betas (!) and the
        // iteration count.
        // This implements a much simpler solution to cancelling the bias by initializing
        // the moving average from the gradients in the first iteration, instead of zero.
        // This avoids the bias entirely.
        // We avoid an iteration count or separate boolean flag by initializing the variance
        // part of the moments to a negative number; in normal operation, the variance can
        // only be positive, and we know we're starting from fresh initialization if the
        // variance is negative.
        // bool isFirstIteration = state.variance < T(0.0f);
        // T blendMean = isFirstIteration ? T(0.0f) : beta1;
        // T blendVariance = isFirstIteration ? T(0.0f) : beta2;
        // state.mean = lerp(grad, state.mean, blendMean);
        // state.variance = lerp(grad * grad, state.variance, blendVariance);
        // param -= learningRate * state.mean / (sqrt(state.variance) + epsilon);
        // grad = T(0.0f);
        state.iteration++;
        state.mean = beta1 * state.mean + (T(1.f) - beta1) * grad;
        state.variance = beta2 * state.variance + (T(1.f) - beta2) * grad * grad;
        T meanHat = state.mean / (T(1.f) - pow(beta1, T(state.iteration)));
        T varianceHat = state.variance / (T(1.f) - pow(beta2, T(state.iteration)));
        param -= learningRate * meanHat / (sqrt(max(T(0.f), varianceHat)) + epsilon);
        grad = T(0.f);
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
