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
}
