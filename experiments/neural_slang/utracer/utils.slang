// SPDX-License-Identifier: Apache-2.0

// Minimal math, ray, frame, sampling, RNG, and lat-long helpers.

import slangpy;

public static const float M_PI = 3.14159265358979323846;
public static const float M_INV_PI = 0.31830988618379067154;
public static const float M_INV_TWO_PI = 0.15915494309189533577;

public float squaredNorm(float2 x) { return dot(x, x); }
public float safeSqrt(float x) { return sqrt(max(x, 0.0)); }

public float3 tonemapAcesFilm(float3 x)
{
    x *= 0.6;
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

public struct Ray
{
    public float3 origin;
    public float tMin;
    public float3 dir;
    public float tMax;

    public __init(float3 origin, float3 dir, float tMin = 0.0, float tMax = 1e30)
    {
        this.origin = origin;
        this.dir = dir;
        this.tMin = tMin;
        this.tMax = tMax;
    }

    public RayDesc toRayDesc() { return { origin, tMin, dir, tMax }; }
};

public float3 offsetRayOrigin(float3 pos, float3 normal)
{
    const float origin = 1.0 / 16.0;
    const float floatScale = 3.0 / 65536.0;
    const float intScale = 3.0 * 256.0;

    int3 offset = int3(normal * intScale);
    float3 posI = float3(
        asfloat(asint(pos.x) + select(pos.x < 0.0, -offset.x, offset.x)),
        asfloat(asint(pos.y) + select(pos.y < 0.0, -offset.y, offset.y)),
        asfloat(asint(pos.z) + select(pos.z < 0.0, -offset.z, offset.z)));
    float3 offsetF = normal * floatScale;
    return float3(
        select(abs(pos.x) < origin, pos.x + offsetF.x, posI.x),
        select(abs(pos.y) < origin, pos.y + offsetF.y, posI.y),
        select(abs(pos.z) < origin, pos.z + offsetF.z, posI.z));
}

public struct Frame
{
    public float3 normal;
    public float3 tangent;
    public float3 bitangent;

    public __init() { normal = 0; tangent = 0; bitangent = 0; }

    public __init(float3 normal, float3 tangent)
    {
        this.normal = normal;
        this.bitangent = normalize(cross(normal, tangent));
        this.tangent = cross(bitangent, normal);
    }

    public float3 toLocal(float3 w)
    {
        return float3(dot(tangent, w), dot(bitangent, w), dot(normal, w));
    }

    public float3 toGlobal(float3 w)
    {
        return w.x * tangent + w.y * bitangent + w.z * normal;
    }
};

public struct UniformSampleGenerator
{
    public uint state;

    public __init(uint2 pixel, uint sampleIndex)
    {
        state = sampleIndex;
        nextUint();
        state ^= pixel.x;
        nextUint();
        state ^= pixel.y;
        nextUint();
    }

    [mutating]
    public uint nextUint()
    {
        state = 1664525u * state + 1013904223u;
        return state;
    }
};

public float nextFloat(inout UniformSampleGenerator sg)
{
    return (sg.nextUint() >> 8) * 0x1p-24;
}

public float2 nextFloat2(inout UniformSampleGenerator sg)
{
    return float2(nextFloat(sg), nextFloat(sg));
}

public float4 nextFloat4(inout UniformSampleGenerator sg)
{
    return float4(nextFloat(sg), nextFloat(sg), nextFloat(sg), nextFloat(sg));
}

public float2 squareToUniformDiskConcentric(float2 sample)
{
    float2 t = 2.0 * sample - 1.0;
    bool isZero = all(t == 0.0);
    bool swapAxes = abs(t.x) < abs(t.y);
    float r = select(swapAxes, t.y, t.x);
    float rp = select(swapAxes, t.x, t.y);
    float phi = 0.25 * M_PI * rp / r;
    phi = select(swapAxes, 0.5 * M_PI - phi, phi);
    phi = select(isZero, 0.0, phi);
    float s, c;
    sincos(phi, s, c);
    return float2(c, s) * r;
}

public float3 squareToCosineHemisphere(float2 sample)
{
    float2 p = squareToUniformDiskConcentric(sample);
    return float3(p, safeSqrt(1.0 - squaredNorm(p)));
}

public float cosineHemispherePdf(float3 p) { return M_INV_PI * p.z; }

public float2 worldToLatlong(float3 dir)
{
    dir = normalize(dir);
    return float2(
        atan2(dir.x, -dir.z) * M_INV_TWO_PI + 0.5,
        acos(dir.y) * M_INV_PI);
}
