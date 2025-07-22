// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing NeuralNetworks;

#include "IScalarActivation.slang"
__include basetypes.DiffCoopVec;

namespace Activation
{

public struct ReLU<T : IReal, int Width> : IScalarActivation<T, Width>
{
    [BackwardDifferentiable, PreferRecompute]
    T activate(T x)
    {
        return max(x, T(0.0f));
    }
};

SCALAR_ACTIVATION_EXTENSION(ReLU)

public extension<T : IReal, int Width> ReLU<T, Width> : IModel<DiffCoopVec<T, Width>, DiffCoopVec<T, Width>>
{
    [BackwardDifferentiable]
    public DiffCoopVec<T, Width> forward(DiffCoopVec<T, Width> x)
    {
        return max(x, DiffCoopVec<T, Width>(0.0f));
    }
}

}
