// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    public void step(inout State state, inout T param, inout T grad);
}

public extension<T : IReal, Optimizer : IOptimizer<T>> Optimizer
{
    public struct Batch
    {
        RWStructuredBuffer<T> params;
        RWStructuredBuffer<T> grads;
        RWStructuredBuffer<Optimizer::State> states;
    }

    public void batch_step<int N>(Batch[N] batches, int2 batch_index)
    {
        Batch batch = batches[batch_index.x];

        int i = batch_index.y;

        T param = batch.params[i];
        T grad = batch.grads[i];
        Optimizer::State state = batch.states[i];

        step(state, param, grad);

        batch.params[i] = param;
        batch.grads[i] = grad;
        batch.states[i] = state;
    }
}
