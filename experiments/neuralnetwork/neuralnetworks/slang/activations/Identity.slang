// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing NeuralNetworks;

#include "IScalarActivation.slang"
__include basetypes.DiffCoopVec;

namespace Activation
{

public struct Identity<T : IReal, int Width> : IScalarActivation<T, Width>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return x;
    }
};

SCALAR_ACTIVATION_EXTENSION(Identity)

public extension<T : IReal, int Width> Identity<T, Width> : IModel<DiffCoopVec<T, Width>, DiffCoopVec<T, Width>>
{
    [BackwardDifferentiable]
    public DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        return x;
    }
}

}
