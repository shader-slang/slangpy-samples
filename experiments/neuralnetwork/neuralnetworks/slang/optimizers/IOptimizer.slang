// SPDX-License-Identifier: Apache-2.0

implementing NeuralNetworks;

__include basetypes.IReal;

/*
    Internal (per-parameter) optimizer state, e.g. Adam moments.
    Has to be constructible from an initial parameter value.
*/
public interface IOptimizerState<T : IReal>
{
    public __init(T param);
}

/*
    Base interface for all optimizers.
    Given a parameter and gradient of precision T, updates the parameter and gradients
    as well as an optional (per-parameter) internal state.
*/
public interface IOptimizer<T : IReal>
{
    public associatedtype State : IOptimizerState<T>;
    public associatedtype Batch;

    public void step(inout State state, inout T param, inout T grad);
    public void batch_step<let N : int>(Batch[N] batches, int2 batch_index);
}
