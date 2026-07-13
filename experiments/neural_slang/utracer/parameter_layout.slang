// SPDX-License-Identifier: Apache-2.0

// Convert the portable row-major checkpoint into the tiled layout consumed by
// WaveTangledVector, and back again when saving a checkpoint.

import model;

public void parametersToOptimalLayout(
    uint index,
    uint threadCount,
    float* portableParameters,
    float* optimalParameters)
{
    NeuralAddress portable = NeuralAddress(portableParameters);
    NeuralAddress optimal = NeuralAddress(optimalParameters);

    NeuralEncoderParameterConverter.toOptimalLayout<NeuralAddress, NeuralAddress>(
        portable,
        optimal,
        index,
        threadCount);
    NeuralDecoderParameterConverter.toOptimalLayout<NeuralAddress, NeuralAddress>(
        portable.getOffset(NeuralEncoderParameterConverter.ElementCountLogical),
        optimal.getOffset(NeuralEncoderParameterConverter.ElementCountPhysical),
        index,
        threadCount);
}

public void parametersToPortableLayout(
    uint index,
    uint threadCount,
    float* optimalParameters,
    float* portableParameters)
{
    NeuralAddress optimal = NeuralAddress(optimalParameters);
    NeuralAddress portable = NeuralAddress(portableParameters);

    NeuralEncoderParameterConverter.toPortableLayout<NeuralAddress, NeuralAddress>(
        optimal,
        portable,
        index,
        threadCount);
    NeuralDecoderParameterConverter.toPortableLayout<NeuralAddress, NeuralAddress>(
        optimal.getOffset(NeuralEncoderParameterConverter.ElementCountPhysical),
        portable.getOffset(NeuralEncoderParameterConverter.ElementCountLogical),
        index,
        threadCount);
}
