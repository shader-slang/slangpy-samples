// SPDX-License-Identifier: Apache-2.0

import main;
import tinymlp;

public static const float3 cameraPosition = float3(0.0f, 0.0f, -5.0f);
public static const float3 lightPosition = float3(0.0f, 0.0f, -3.0f);
public static const int MAX_STEPS = 128;    // Max ray marching steps
public static const float MAX_DIST = 100.0; // Maximum render distance
public static const float SURFACE_DIST = 0.001; // Surface hit precision

public struct Ray
{
    public float3 origin;
    public float3 direction;
};

public interface ISDFModel
{
    public float sdfFunc(float3 p);
    public float3 estimateNormal(float3 p);
    public float3 getObjColor();
}

public float targetSDF(no_diff float3 samplePoint)
{
    return length(samplePoint) - 2.0f;
}

// This model uses a learned MLP to approximate the SDF.
public struct LearnedSDFModle : ISDFModel
{
    public TinyMLP mlp;
    public float3 color = float3(1.0, 0.6, 0.3);

    public __init(TinyMLP inMlp, float3 inColor)
    {
        mlp = inMlp;
        color = inColor;
    }

    public float sdfFunc(float3 p)
    {
        // Since the MLP actually learns the function x^2 + y^2 + z^2, we need to convert it to the SDF function.
        // The SDF function is sqrt(x^2 + y^2 + z^2) - 2.0f.
        float[input_size] input = {p.x, p.y, p.z};
        float result = mlp.eval(input)[0];
        result = (result < 0.0f) ? 0.0f : sqrt(result);
        result = result - 2.0f;
        return (result < 0.0f) ? 0.0f : result;
    }

    public float3 estimateNormal(float3 p)
    {
        return normalize(p);
    }

    public float3 getObjColor()
    {
        return color;
    }
}

// The target SDF model is a sphere with radius 2.0f.
public struct TargetSDFModle : ISDFModel
{
    public float3 color = float3(0.5, 0.6, 1.0);

    public __init(float3 inColor)
    {
        color = inColor;
    }

    public float sdfFunc(float3 p)
    {
        return length(p) - 2.0f; // radius - 2.0f
    }

    public float3 estimateNormal(float3 p)
    {
        return normalize(p);
    }

    public float3 getObjColor()
    {
        return color;
    }
}

public struct RayMarcher
{
    public Ray ray;
    public float3 getModelColor(ISDFModel sdfModel)
    {
        float dist = march(sdfModel);

        float3 color = float3(0.0);
        if (dist > 0.0)
        {
            float3 hitPoint = ray.origin + ray.direction * dist;
            float3 normal = sdfModel.estimateNormal(hitPoint);
            float3 objColor = sdfModel.getObjColor();
            color = objColor * computeLighting(hitPoint, normal);
        }

        return color;
    }

    float march(ISDFModel sdfModel)
    {
        float totalDist = 0.0;
        for (int i = 0; i < MAX_STEPS; i++)
        {
            float3 p = ray.origin + ray.direction * totalDist;
            float dist = sdfModel.sdfFunc(p);

            if (dist < SURFACE_DIST)
                return totalDist; // Hit the surface

            if (totalDist > MAX_DIST)
                return -1.0;     // Exceeded max distance

            totalDist += dist;
        }
        return -1.0; // No hit
    }

    public float computeLighting(float3 p, float3 normal)
    {
        float3 lightDir = normalize(lightPosition - p);
        float diffuse = max(dot(normal, lightDir), 0.0);
        return diffuse;
    }
}
