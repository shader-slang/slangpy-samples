// SPDX-License-Identifier: Apache-2.0

// Neural-only training path. The MDL ceramic material is used solely as the
// teacher that produces encoder features and reference BSDF values.

import slangpy;
import model;
import materials.materials;
import materials.ceramic.mdl_ceramic_material;
import utracerutils;

static const float kMinCos = 1e-3f;

static void packLayer(
    inout float[kEncoderInputs] inputs,
    inout int index,
    float3 normal,
    float3 tangent,
    float3 albedo,
    float2 roughness,
    float3 weight)
{
    inputs[index++] = normal.x;
    inputs[index++] = normal.y;
    inputs[index++] = normal.z;
    inputs[index++] = tangent.x;
    inputs[index++] = tangent.y;
    inputs[index++] = tangent.z;
    inputs[index++] = albedo.x;
    inputs[index++] = albedo.y;
    inputs[index++] = albedo.z;
    inputs[index++] = roughness.x;
    inputs[index++] = roughness.y;
    inputs[index++] = weight.x;
    inputs[index++] = weight.y;
    inputs[index++] = weight.z;
}

public float[kEncoderInputs] collectEncoderInputs(float2 uv)
{
    float[kEncoderInputs] inputs = {};
    int index = 0;

    ShadingData sd;
    sd.position = float3(uv, 0);
    sd.view = float3(0, 0, 1);
    sd.uv = uv;
    sd.shading_frame = Frame(float3(0, 0, 1), float3(1, 0, 0));
    sd.geometric_normal = sd.shading_frame.normal;
    sd.geometric_tangent = float4(sd.shading_frame.tangent, 1);
    sd.front_facing = true;

    let material = MDLCeramicMaterialInstance(sd, true);
    Optional<ExtraBsdfProperties> extra =
        material.collect_extra_bsdf_properties(sd, sd.view);

    if (extra.hasValue && extra.value.bsdf_count > 0)
    {
        ExtraBsdfProperties props = extra.value;
        [ForceUnroll]
        for (int layer = 0; layer < kNumBsdfLayers; ++layer)
        {
            if (layer < int(props.bsdf_count))
            {
                packLayer(
                    inputs,
                    index,
                    props.bsdf_N[layer],
                    props.bsdf_T[layer],
                    props.bsdf_albedo[layer],
                    props.bsdf_roughness[layer],
                    props.bsdf_weight[layer]);
            }
            else
            {
                packLayer(
                    inputs, index, float3(0), float3(0), float3(0),
                    float2(0), float3(0));
            }
        }
    }
    else
    {
        MaterialProperties props = material.collect_properties(sd);
        packLayer(
            inputs,
            index,
            props.guide_normal,
            sd.shading_frame.tangent,
            props.diffuse_reflection_albedo + props.specular_reflection_albedo,
            float2(props.roughness),
            float3(1.0));
        packLayer(
            inputs, index, float3(0), float3(0), float3(0),
            float2(0), float3(0));
    }

    // The original example uses one fixed mip and stores 1 + mip_level.
    inputs[index] = 1.0;
    return inputs;
}

[Differentiable]
static float3 sphericalToCartesian(float2 spherical)
{
    return float3(
        -cos(spherical.y - M_PI) * sin(spherical.x),
        -sin(spherical.y - M_PI) * sin(spherical.x),
        cos(spherical.x));
}

[Differentiable]
static float3 rotateY(float3 v, float angle)
{
    float s, c;
    sincos(angle, s, c);
    return float3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

[Differentiable]
static float3 rotateZ(float3 v, float angle)
{
    float s, c;
    sincos(angle, s, c);
    return float3(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
}

[Differentiable]
static void rusinkiewiczToCartesian(
    float2 halfSpherical,
    float2 differenceSpherical,
    out float3 wi,
    out float3 wo)
{
    float3 difference = sphericalToCartesian(differenceSpherical);
    wi = rotateZ(rotateY(difference, halfSpherical.x), halfSpherical.y);
    float3 halfVector = sphericalToCartesian(halfSpherical);
    wo = normalize(2.0 * dot(wi, halfVector) * halfVector - wi);
}

static bool sampleUniformRusinkiewicz(float4 u, out float3 wi, out float3 wo)
{
    float thetaH = u.x * M_PI * 0.5;
    float thetaD = u.y * M_PI * 0.5;
    float phiH = u.z * M_PI * 2.0;
    float phiD = u.w * M_PI * 2.0;
    rusinkiewiczToCartesian(float2(thetaH, phiH), float2(thetaD, phiD), wi, wo);
    return wi.z >= kMinCos && wo.z >= kMinCos;
}

static float3 evalReference(
    float2 uv,
    float3 wi,
    float3 wo,
    UniformSampleGenerator sg)
{
    ShadingData sd;
    sd.position = float3(uv, 0);
    sd.view = wi;
    sd.uv = uv;
    sd.shading_frame = Frame(float3(0, 0, 1), float3(1, 0, 0));
    sd.geometric_normal = sd.shading_frame.normal;
    sd.geometric_tangent = float4(sd.shading_frame.tangent, 1);
    sd.front_facing = true;
    let material = MDLCeramicMaterialInstance(sd, true);
    return material.eval(sd, wo, sg);
}

[Differentiable]
static float logTransform(float x)
{
    // Stable compression for HDR BSDF values. Unlike a cube root, log1p has a
    // finite derivative at zero-valued material channels.
    return log(1.0 + max(x, 0.0));
}

[Differentiable]
static float trainingLoss(
    NeuralAddress parameters,
    no_diff NeuralNetwork network,
    NeuralEncoderInputVector encoderInput,
    float3 wi,
    float3 wo,
    no_diff float3 reference)
{
    float3 prediction = network.evalVector<NeuralAddress>(
        encoderInput, wi, wo, parameters);

    float loss = 0.0;
    [ForceUnroll]
    for (int i = 0; i < 3; ++i)
        loss += abs(logTransform(prediction[i]) - logTransform(reference[i]));
    return loss / 3.0;
}

static uint3 colorAugmentationIndices(float ratio, float sampleValue)
{
    if (ratio > 0.0 && sampleValue < ratio)
    {
        float r = 27.0 * sampleValue / ratio;
        return uint3(uint(r / 9.0), uint(r / 3.0), uint(r)) % 3;
    }
    return uint3(0, 1, 2);
}

/// Evaluate the MDL teacher and store a compact batch. Keeping the large MDL
/// evaluation outside the reverse-mode kernel avoids retaining its local
/// state on the neural autodiff tape.
public void prepareTrainingSample(
    uint seed,
    int2 batchIndex,
    int2 batchSize,
    no_diff float colorAugmentationRatio,
    WTensor<float, 3> encoderInputOutput,
    WTensor<float, 3> wiOutput,
    WTensor<float, 3> woOutput,
    WTensor<float, 3> referenceOutput,
    WTensor<float, 2> validOutput)
{
    UniformSampleGenerator sg = UniformSampleGenerator(batchIndex, seed);
    float2 uv = (float2(batchIndex) + next_float2(sg)) / float2(batchSize);

    float3 wi, wo;
    bool valid = sampleUniformRusinkiewicz(next_float4(sg), wi, wo);
    float3 reference = valid ? evalReference(uv, wi, wo, sg) : float3(0.0);
    valid = valid && all(!isnan(reference)) && all(!isinf(reference));

    uint3 indices = colorAugmentationIndices(
        colorAugmentationRatio, next_float(sg));
    reference = float3(
        reference[indices.x], reference[indices.y], reference[indices.z]);

    float[kEncoderInputs] encoderInputs = collectEncoderInputs(uv);
    [ForceUnroll]
    for (int i = 0; i < kEncoderInputs; ++i)
    {
        bool finite = !isnan(encoderInputs[i]) && !isinf(encoderInputs[i]);
        valid = valid && finite;
        encoderInputs[i] = finite ? encoderInputs[i] : 0.0;
        encoderInputOutput[batchIndex.y, batchIndex.x, i] = encoderInputs[i];
    }

    float3 safeWi = valid ? wi : float3(0, 0, 1);
    float3 safeWo = valid ? wo : float3(0, 0, 1);
    reference = valid ? reference : float3(0.0);
    [ForceUnroll]
    for (int i = 0; i < 3; ++i)
    {
        wiOutput[batchIndex.y, batchIndex.x, i] = safeWi[i];
        woOutput[batchIndex.y, batchIndex.x, i] = safeWo[i];
        referenceOutput[batchIndex.y, batchIndex.x, i] = reference[i];
    }
    validOutput[batchIndex.y, batchIndex.x] = valid ? 1.0 : 0.0;
}

/// Accumulate one prepared sample's neural gradients per invocation.
/// There are deliberately no early returns: WaveTangledVector requires all
/// 32 lanes to reach every neural layer's workgroup barriers.
public void calculateGradients(
    int2 batchIndex,
    RWTensor<float, 3> encoderInput,
    RWTensor<float, 3> wiInput,
    RWTensor<float, 3> woInput,
    RWTensor<float, 3> referenceInput,
    RWTensor<float, 2> validInput,
    float* parameters,
    float* parameterGradients)
{
    float[kEncoderInputs] encoderInputs = {};
    [ForceUnroll]
    for (int i = 0; i < kEncoderInputs; ++i)
        encoderInputs[i] = encoderInput[batchIndex.y, batchIndex.x, i];

    float3 wi, wo, reference;
    [ForceUnroll]
    for (int i = 0; i < 3; ++i)
    {
        wi[i] = wiInput[batchIndex.y, batchIndex.x, i];
        wo[i] = woInput[batchIndex.y, batchIndex.x, i];
        reference[i] = referenceInput[batchIndex.y, batchIndex.x, i];
    }

    NeuralNetwork network = NeuralNetwork();
    var encoderInputPair = diffPair(NeuralEncoderInputVector(encoderInputs));
    var wiPair = diffPair(wi);
    var woPair = diffPair(wo);

    var parameterPair = DifferentialPtrPair<NeuralAddress>(
        NeuralAddress(parameters), NeuralAddress(parameterGradients));
    bwd_diff(trainingLoss)(
        parameterPair,
        network,
        encoderInputPair,
        wiPair,
        woPair,
        reference,
        validInput[batchIndex.y, batchIndex.x]);
}

/// Deterministic validation loss, one output value per invocation.
public void computeLoss(
    uint seed,
    int2 batchIndex,
    int2 batchSize,
    float* parameters,
    WTensor<float, 2> output)
{
    UniformSampleGenerator sg = UniformSampleGenerator(batchIndex, seed);
    float2 uv = (float2(batchIndex) + next_float2(sg)) / float2(batchSize);

    float3 wi, wo;
    bool valid = sampleUniformRusinkiewicz(next_float4(sg), wi, wo);
    float3 reference = valid ? evalReference(uv, wi, wo, sg) : float3(0.0);
    valid = valid && all(!isnan(reference)) && all(!isinf(reference));

    float3 safeWi = valid ? wi : float3(0, 0, 1);
    float3 safeWo = valid ? wo : float3(0, 0, 1);
    float[kEncoderInputs] encoderInputs = collectEncoderInputs(uv);

    NeuralNetwork network = NeuralNetwork();
    float3 prediction = network.eval<NeuralAddress>(
        encoderInputs, safeWi, safeWo, NeuralAddress(parameters));

    float loss = 0.0;
    [ForceUnroll]
    for (int i = 0; i < 3; ++i)
        loss += abs(logTransform(prediction[i]) - logTransform(reference[i]));
    output[batchIndex.y, batchIndex.x] = valid ? loss / 3.0 : -1.0;
}

/// Bake the learned encoder into an HxWx8 latent tensor.
public void bakeLatentTexture(
    int2 pixel,
    int2 resolution,
    float* parameters,
    WTensor<float, 3> latentTexture)
{
    float2 uv = (float2(pixel) + 0.5) / float2(resolution);
    float[kEncoderInputs] encoderInputs = collectEncoderInputs(uv);
    NeuralNetwork network = NeuralNetwork();
    NeuralLatentVector latents = network.encoder.eval<NeuralAddress>(
        NeuralEncoderInputVector(encoderInputs), NeuralAddress(parameters));

    [ForceUnroll]
    for (int i = 0; i < kNumLatents; ++i)
        latentTexture[pixel.y, pixel.x, i] = latents[i];
}

public void optimizerStep(
    int index,
    float* parameters,
    float* parameterGradients,
    float* firstMoment,
    float* secondMoment,
    float learningRate,
    int iteration)
{
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    const float epsilon = 1e-8;

    float gradient = parameterGradients[index];
    if (isnan(gradient) || isinf(gradient))
        gradient = 0.0;

    float mean = lerp(gradient, firstMoment[index], beta1);
    float variance = lerp(gradient * gradient, secondMoment[index], beta2);
    firstMoment[index] = mean;
    secondMoment[index] = variance;

    float meanHat = mean / (1.0 - pow(beta1, iteration));
    float varianceHat = variance / (1.0 - pow(beta2, iteration));
    float value = parameters[index] -
        learningRate * meanHat / (sqrt(varianceHat) + epsilon);
    if (!isnan(value) && !isinf(value))
        parameters[index] = value;
    parameterGradients[index] = 0.0;
}
