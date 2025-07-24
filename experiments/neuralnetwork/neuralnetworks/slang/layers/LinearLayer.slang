// SPDX-License-Identifier: Apache-2.0

implementing NeuralNetworks;

__include basetypes.IModel;
__include basetypes.DiffCoopVec;

import slangpy;

/*
    Represents a linear neural network layer, i.e. a matrix-vector multiply/add A * x + b

    Takes an input with NumInputs elements, and returns NumOutputs elements.

    LinearLayer implements the multiply in software for plain arrays,
    and CoopVecLinearLayer uses hardware acceleration using cooperative vectors.
*/
public struct LinearLayer<
    T : IReal,
    int NumInputs,
    int NumOutputs
> : IModel<T[NumInputs], T[NumOutputs]> where T.Differential : IAtomicAddable
{
    public GradOutTensor<T, 2> weights;
    public GradOutTensor<T, 1> biases;

    [BackwardDerivative(backward)]
    public OutputType forward(InputType x)
    {
        OutputType y;
        for (int row = 0; row < NumOutputs; ++row)
        {
            var sum = biases[row];
            [ForceUnroll]
            for (int col = 0; col < NumInputs; ++col)
                sum += weights[row, col] * x[col];

            y[row] = sum;
        }

        return y;
    }

    public void backward(inout DifferentialPair<InputType> x, OutputType.Differential grad)
    {
        var dx = InputType.dzero();
        for (int row = 0; row < NumOutputs; ++row)
        {
            biases.d_out.set({row}, grad[row]);

            [ForceUnroll]
            for (int col = 0; col < NumInputs; ++col)
            {
                weights.d_out.set({row, col}, T.dmul(x.p[col], grad[row]));
                dx[col] = T.dadd(dx[col], T.dmul(weights[row, col], grad[row]));
            }
        }
        x = diffPair(x.p, dx);
    }
}

public struct CoopVecLinearLayer<
    T : IReal,
    int NumInputs,
    int NumOutputs
> : IModel<DiffCoopVec<T, NumInputs>, DiffCoopVec<T, NumOutputs>> where T.Differential : IAtomicAddable
{
    static const CoopVecComponentType ComponentType = InputType::ComponentType;

    public GradOutTensor<T, 1> weights;
    public GradOutTensor<T, 1> biases;

    [BackwardDerivative(backward)]
    public OutputType forward(InputType x)
    {
        return OutputType(coopVecMatMulAdd<T, NumOutputs>(
            x.cv, ComponentType,
            weights.primal.buffer, 0, ComponentType,
            biases.primal.buffer, 0, ComponentType,
            CoopVecMatrixLayout.TrainingOptimal, false, 0
        ));
    }

    public void backward(inout DifferentialPair<InputType> x, OutputType grad)
    {
        coopVecOuterProductAccumulate(grad.cv, x.p.cv, weights.d_out.buffer, 0, 0, CoopVecMatrixLayout.TrainingOptimal, ComponentType);
        coopVecReduceSumAccumulate(grad.cv, biases.d_out.buffer, 0);

        let dX = coopVecMatMul<T, NumInputs>(grad.cv, ComponentType, weights.primal.buffer, 0, ComponentType, CoopVecMatrixLayout.TrainingOptimal, true, 0);

        x = diffPair(x.p, InputType(dX));
    }
}
