// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing NeuralNetworks;

#include "IScalarActivation.slang"
__include basetypes.DiffCoopVec;

namespace Activation
{

public struct Swish<T : IReal, int Width> : IScalarActivation<T, Width>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return x / (T(1.0f) + exp(-x));
    }
};

SCALAR_ACTIVATION_EXTENSION(Swish)

public extension<T : IReal, int Width> Swish<T, Width> : IModel<DiffCoopVec<T, Width>, DiffCoopVec<T, Width>>
{
    [BackwardDifferentiable]
    public DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        return x / (1.0f + exp(-x));
    }
}

}
