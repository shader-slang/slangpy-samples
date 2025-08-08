// SPDX-License-Identifier: Apache-2.0

// clang-format off

__exported import NeuralNetworks;
__exported import Utils;

// This is the reference material the network should learn
// We use a textured Disney BRDF for this example
struct TexturedMaterial {
    Texture2D<float4> albedoMap;
    Texture2D<float4> metalRoughMap;
    Texture2D<float4> normalMap;
    SamplerState sampler;

    // Load the BRDF parameters for a given uv coordinate
    DisneyBRDF load(float2 uv)
    {
        float3 albedo = albedoMap.SampleLevel(sampler, uv, 0.0f).rgb;
        float roughness = metalRoughMap.SampleLevel(sampler, uv, 0.0f).r;
        float3 normal = normalMap.SampleLevel(sampler, uv, 0.0f).rgb;

        return {
            albedo,
            normalize(normal - 0.5f),
            roughness,
            0.0f,
            1.0f
        };
    }
}

// This is the input the neural network receives during training
// It receives the incoming/outgoing directions, as well as the parameters of the reference material
struct MaterialTrainInput : IDifferentiable {
    float3 wi;
    float3 wo;
    DisneyBRDF referenceMaterial;
}
// A neural material is a trainable model that takes a MaterialTrainInput and returns an RGB float3
typealias INeuralMaterial = IModel<MaterialTrainInput, float3>;

void trainMaterial<NeuralMaterial : INeuralMaterial>(TexturedMaterial referenceMaterial, NeuralMaterial neuralMaterial, inout RNG rng, float lossScale)
{
    // To train a neural material, we first sample a random UV coordinate and incoming and outgoing direction
    float2 uv = rng.next2D();
    // We sample directions from a simple distribution for this example. For faster training,
    // a better option is e.g. sampling the Rusinkiewicz parametrization suggested by Zeltner et al.
    float3 wi = rng.randomDirection();
    float3 n = normalize(rng.randomDirection() * float3(0.2f, 0.2f, 1.0f));
    float3 wo = reflect(-wi, n);

    // Then we load the reference material and evaluate its color for the given UV/directions
    DisneyBRDF brdf = referenceMaterial.load(uv);
    float3 targetValue = brdf.eval(wi, wo);

    // Finally, we differentiate the loss of the model for the given input and target
    bwd_diff(evalModelAndLoss)(neuralMaterial, {wi, wo, brdf}, targetValue, lossScale);
}

[BackwardDifferentiable]
float evalModelAndLoss<NeuralMaterial : INeuralMaterial>(NeuralMaterial neuralMaterial, no_diff MaterialTrainInput input, no_diff float3 target)
{
    // A material can have high dynamic range, so we are a bit careful here when computing the loss
    float3 prediction = neuralMaterial.forward(input);

    // First, separate the material response into brightness and color
    const float3 luminance = float3(0.2126f, 0.7152f, 0.0722f);
    float predictionLuminance = dot(prediction, luminance);
    float targetLuminance = dot(target, luminance);
    float3 predictionColor = prediction / max(detach(predictionLuminance), 0.05f);
    float3 targetColor = target / max(detach(targetLuminance), 0.05f);

    // Then, compute a simple L2 loss for the color...
    float colorLoss = Losses::L2().eval(predictionColor, targetColor);
    // ...and an L1 loss for the luminance, in a compressed log space
    float luminanceLoss = abs(log(predictionLuminance + 1.0f) - log(targetLuminance + 1.0f));

    // Finally, combine them with empirically chosen weights
    return colorLoss * 3.0f + luminanceLoss;
}

// Implementation of a very simple neural material based on an encoder/decoder architecture,
// loosely inspired by the paper "Real-Time Neural Appearance Models" by Zeltner et al.
// To keep this example simple, we are omitting a few pieces from the paper, such as the shading
// frame encoding, input encodings and LOD, among other things.
// We also run the encoder on every evaluation, even though it is only needed for training.
// After training is complete, you should generate a latent texture by running the encoder
// on every pixel of the reference material. Then the encoder can be dropped, and the latent texture
// can be sampled directly to get the latents to pass to the decoder.
//
// We assume a fixed number of parameters of the reference material, and set the size of the latent
// vector to 8.
static const int NumBRDFParams = 9;
static const int NumLatents = 8;

struct NeuralMaterial<
    EncoderNetwork : IModel<float[NumBRDFParams], float[NumLatents]>,
    DecoderNetwork : IModel<float[6 + NumLatents], float3>
> : INeuralMaterial {

    // These are the nested networks we will use. We use basic MLPs for this example.
    EncoderNetwork encoder;
    DecoderNetwork decoder;

    [BackwardDifferentiable]
    float3 forward(MaterialTrainInput input)
    {
        // First step: Run the encoder to turn the reference material parameters into a latent code
        // Unpack the parameters into a flat array...
        float materialParams[NumBRDFParams];
        materialParams[0] = input.referenceMaterial.baseColor.r;
        materialParams[1] = input.referenceMaterial.baseColor.g;
        materialParams[2] = input.referenceMaterial.baseColor.b;
        materialParams[3] = input.referenceMaterial.normal.r;
        materialParams[4] = input.referenceMaterial.normal.g;
        materialParams[5] = input.referenceMaterial.normal.b;
        materialParams[6] = input.referenceMaterial.roughness;
        materialParams[7] = input.referenceMaterial.metallic;
        materialParams[8] = input.referenceMaterial.specular;

        // ... then run the encoder
        let latents = encoder.forward(materialParams);

        // Second step: Run the decoder to turn the latent code and the directions into a material response
        // Combine the directions and the latent code into a flat array...
        float decoderInput[6 + NumLatents];
        decoderInput[0] = input.wi.x;
        decoderInput[1] = input.wi.y;
        decoderInput[2] = input.wi.z;
        decoderInput[3] = input.wo.x;
        decoderInput[4] = input.wo.y;
        decoderInput[5] = input.wo.z;
        for (int i = 0; i < NumLatents; ++i)
            decoderInput[6 + i] = latents[i];

        // ... then run the decoder
        return decoder.forward(decoderInput);
    }
}

// This function will render the neural material on simple geometry
float4 renderMaterial<NeuralMaterial : INeuralMaterial>(
    TexturedMaterial referenceMaterial,
    NeuralMaterial neuralMaterial,
    int iteration,
    uint2 pixel
) {
    // Some hard coded scene parameters:
    const float3 lightDirection = normalize(float3(0.4f, -0.5f, -0.8f));
    const float3 cameraPos = float3(0.0f, 0.0f, -2.0f);
    const float3 spherePos = float3(0.0f, 0.0f, 0.0f);

    // Assume a 1024x1024 window here
    float2 xy = (pixel - 512.0f) / 512.0f;
    float3 dir = normalize(float3(xy, 1.0f));

    // Then, render a basic sphere
    float t = raytraceSphere(cameraPos, dir, spherePos, 1.3f);
    if (t == 0.0f)
        return float4(0.0f, 0.0f, 0.0f, 1.0f);

    float3 hitPos = cameraPos + dir * t;
    float3 N = normalize(hitPos - spherePos);
    float3 T = normalize(cross(N, float3(0.0f, 1.0f, 0.0f)));
    float3 B = cross(N, T);
    float u = fmod(2.0f * atan2(N.z, N.x) / (2.0f * float.getPi()) + iteration * 0.002f, 1.0f);
    float v = asin(N.y) / float.getPi() + 0.5f;

    // Compute the directions to light and camera and transform them into the shading frame
    float3 wi = -dir;
    float3 wo = lightDirection;
    float3 wiLocal = {dot(wi, T), dot(wi, B), dot(wi, N)};
    float3 woLocal = {dot(wo, T), dot(wo, B), dot(wo, N)};

    // Load the reference material parameters for the given UV coordinate
    DisneyBRDF brdf = referenceMaterial.load(float2(u, v));

    // Finally, evaluate the neural material response if the directions are above the horizon
    var color = float3(0.0f);
    if (woLocal.z > 0 && wiLocal.z > 0)
        color = neuralMaterial.forward({wiLocal, woLocal, brdf});
    return float4(color, 1.0f);
}
