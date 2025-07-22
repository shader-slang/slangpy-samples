// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing NeuralNetworks;

#include "IScalarActivation.slang"
__include basetypes.DiffCoopVec;

namespace Activation
{

public struct Tanh<T : IReal, int Width> : IScalarActivation<T, Width>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return tanh(x);
    }
};

SCALAR_ACTIVATION_EXTENSION(Tanh)

public extension<T : IReal, int Width> Tanh<T, Width> : IModel<DiffCoopVec<T, Width>, DiffCoopVec<T, Width>>
{
    [BackwardDifferentiable]
    public DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        return tanh(x);
    }
}

}
