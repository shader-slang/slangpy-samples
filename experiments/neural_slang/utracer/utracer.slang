// SPDX-License-Identifier: Apache-2.0

// Neural-only micro path tracer. There is no analytic or MDL render branch:
// every surface hit uses the baked encoder latents and neural.slang decoder.

import slangpy;
import utils;
import model;

public struct Vertex
{
    public float3 position;
    public float3 normal;
    public float3 tangent;
    public float2 uv;
};

public struct VertexData
{
    public float3 position;
    public float3 normal;
    public float3 tangent;
    public float3 faceNormal;
    public float2 uv;
};

public struct HitInfo
{
    public bool valid;
    public float2 barycentrics;
    public float t;
    public uint primitiveIndex;
};

public struct Camera
{
    public float3 position;
    public float3 imageU;
    public float3 imageV;
    public float3 imageW;
    public int2 dimensions;

    public Ray getRay(float2 uv)
    {
        uv = uv * 2.0 - 1.0;
        return Ray(position, normalize(uv.x * imageU + uv.y * imageV + imageW));
    }
};

public struct EnvironmentMap
{
    Texture2D<float4> texture;
    float3 scale;
    bool valid;

    public float3 eval(float3 direction, SamplerState sampler)
    {
        return texture.SampleLevel(sampler, worldToLatlong(direction), 0).xyz * scale;
    }
};

public struct BakedNeuralMaterial
{
    RWTensor<float, 3> latentTexture;

    public NeuralLatentVector sampleLatents(float2 uv)
    {
        int2 size = int2(latentTexture.shape[1], latentTexture.shape[0]);
        float2 position = saturate(uv) * float2(size - 1);
        int2 p0 = clamp(int2(position), int2(0), size - 1);
        int2 p1 = min(p0 + 1, size - 1);
        float2 f = saturate(position - float2(p0));

        float[kNumLatents] values = {};
        [ForceUnroll]
        for (int i = 0; i < kNumLatents; ++i)
        {
            float v00 = latentTexture[p0.y, p0.x, i];
            float v10 = latentTexture[p0.y, p1.x, i];
            float v01 = latentTexture[p1.y, p0.x, i];
            float v11 = latentTexture[p1.y, p1.x, i];
            values[i] = lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
        }
        return NeuralLatentVector(values);
    }
};

public struct Scene
{
    RaytracingAccelerationStructure tlas;
    StructuredBuffer<Vertex> vertices;
    StructuredBuffer<uint> indices;
    BakedNeuralMaterial material;
    SamplerState sampler;
    EnvironmentMap environment;

    public VertexData getVertexData(HitInfo hit)
    {
        uint i0 = indices[hit.primitiveIndex * 3 + 0];
        uint i1 = indices[hit.primitiveIndex * 3 + 1];
        uint i2 = indices[hit.primitiveIndex * 3 + 2];
        Vertex v0 = vertices[i0];
        Vertex v1 = vertices[i1];
        Vertex v2 = vertices[i2];
        float3 b = float3(
            1.0 - hit.barycentrics.x - hit.barycentrics.y,
            hit.barycentrics.x,
            hit.barycentrics.y);

        VertexData result;
        result.position = b.x * v0.position + b.y * v1.position + b.z * v2.position;
        result.normal = normalize(b.x * v0.normal + b.y * v1.normal + b.z * v2.normal);
        result.tangent = normalize(b.x * v0.tangent + b.y * v1.tangent + b.z * v2.tangent);
        result.faceNormal = normalize(cross(v1.position - v0.position, v2.position - v0.position));
        result.uv = b.x * v0.uv + b.y * v1.uv + b.z * v2.uv;
        return result;
    }
};

public ParameterBlock<Scene> gScene;
uniform float* gNeuralParameters;

static HitInfo intersect(Ray ray)
{
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(gScene.tlas, 0, 0xff, ray.toRayDesc());
    query.Proceed();

    HitInfo hit = {};
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        hit.valid = true;
        hit.barycentrics = query.CommittedTriangleBarycentrics();
        hit.t = query.CommittedRayT();
        hit.primitiveIndex = query.CommittedPrimitiveIndex();
    }
    return hit;
}

static const uint kMaxDepth = 5;

/// Fixed-depth, lockstep path traversal.
///
/// WaveTangledVector uses workgroup barriers internally. Even after a ray has
/// missed, its lane therefore continues with harmless dummy decoder inputs so
/// every lane reaches exactly one neural evaluation at every bounce.
static float3 tracePath(Ray ray, inout UniformSampleGenerator sg)
{
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    bool active = true;
    NeuralNetwork network = NeuralNetwork();
    NeuralAddress parameters = NeuralAddress(gNeuralParameters);

    [ForceUnroll]
    for (uint depth = 0; depth < kMaxDepth; ++depth)
    {
        HitInfo hit = {};
        if (active)
            hit = intersect(ray);

        bool missed = active && !hit.valid;
        if (missed)
        {
            if (gScene.environment.valid)
                radiance += throughput * gScene.environment.eval(ray.dir, gScene.sampler);
            active = false;
        }

        bool hasSurface = active && hit.valid;
        VertexData vertex = {};
        Frame frame = Frame();
        if (hasSurface)
        {
            vertex = gScene.getVertexData(hit);
            frame = Frame(vertex.normal, vertex.tangent);
        }

        // Every lane consumes the same number of random values and enters the
        // decoder, including lanes whose paths have already terminated.
        float3 sampledWo = squareToCosineHemisphere(nextFloat2(sg));
        float pdf = cosineHemispherePdf(sampledWo);
        float3 localWi = hasSurface ? frame.toLocal(-ray.dir) : float3(0, 0, 1);
        bool canShade = hasSurface && localWi.z > 1e-3 && pdf > 1e-3;
        float3 safeWi = canShade ? localWi : float3(0, 0, 1);
        float3 safeWo = canShade ? sampledWo : float3(0, 0, 1);
        float2 safeUv = canShade ? vertex.uv : float2(0.0);

        NeuralLatentVector latents = gScene.material.sampleLatents(safeUv);
        float3 weight = network.evalBaked<NeuralAddress>(
            latents, safeWi, safeWo, parameters) / max(pdf, 1e-3);

        if (canShade)
        {
            bool finite = all(!isnan(weight)) && all(!isinf(weight));
            if (finite)
            {
                throughput *= clamp(weight, 0.0, 128.0);
                ray = Ray(
                    offsetRayOrigin(vertex.position, vertex.faceNormal),
                    frame.toGlobal(sampledWo));
            }
            else
            {
                active = false;
            }
        }
        else if (hasSurface)
        {
            active = false;
        }
    }
    return radiance;
}

public extension Camera
{
    public void __slangpy_load(Context2D context, out RaySampler value)
    {
        value.camera = this;
        value.id = context.call_id_vec;
        value.pixelSize = 1.0 / this.dimensions;
        value.uv = float2(context.call_id_vec) / this.dimensions;
        value.uv.y = 1.0 - value.uv.y;
    }

    public void __slangpy_store(Context2D context, in RaySampler value) {}
}

public struct RaySampler
{
    public Camera camera;
    public uint2 id;
    public float2 uv;
    public float2 pixelSize;

    public Ray sampleRay(inout UniformSampleGenerator sg)
    {
        float2 offset = nextFloat2(sg) * float2(1, -1);
        return camera.getRay(uv + offset * pixelSize);
    }
};

public void render(
    RaySampler raySampler,
    out float4 output,
    uniform uint samplesPerPixel,
    uniform uint iteration)
{
    float3 radiance = 0.0;
    for (uint sample = 0; sample < samplesPerPixel; ++sample)
    {
        UniformSampleGenerator sg = UniformSampleGenerator(
            raySampler.id, iteration * samplesPerPixel + sample);
        radiance += tracePath(raySampler.sampleRay(sg), sg);
    }
    output = float4(radiance / max(float(samplesPerPixel), 1.0), 1.0);
}

public void accumulator_reset(out float4 accumulator) { accumulator = 0.0; }

public void accumulator_output(float4 accumulator, out float4 output)
{
    output = float4(accumulator.xyz / max(accumulator.w, 1.0), 1.0);
}

public void accumulator_update(float4 input, inout float4 accumulator)
{
    accumulator += float4(input.xyz, 1.0);
}

public void resample_rgb(
    uint2 dispatchThreadID,
    Texture2D<float4> input,
    SamplerState inputSampler,
    RWTexture2D<float4> output,
    uniform uint2 outputSize)
{
    uint2 dimensions;
    output.GetDimensions(dimensions.x, dimensions.y);
    if (any(dispatchThreadID >= dimensions) || any(dispatchThreadID >= outputSize))
        return;
    float2 uv = (float2(dispatchThreadID) + 0.5) / float2(outputSize);
    float3 color = input.SampleLevel(inputSampler, uv, 0).xyz;
    output[dispatchThreadID] = float4(tonemapAcesFilm(color), 1.0);
}
