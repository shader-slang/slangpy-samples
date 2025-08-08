// SPDX-License-Identifier: Apache-2.0

struct RNG
{
    uint state;

    __init(uint state) { this.state = state; }

    [mutating]
    uint next()
    {
        state = state * 2739110765U + 2739110765U;
        return state;
    }

    [mutating]
    float next1D()
    {
        // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
        // In floating-point precision this also ensures that 1.0-u != 0.0.
        uint bits = next();
        return (bits >> 8) * 0x1p-24;
    }

    [mutating]
    float2 next2D()
    {
        return float2(next1D(), next1D());
    }

    [mutating]
    float3 randomDirection()
    {
        float r = sqrt(next1D());
        float phi = next1D() * float.getPi() * 2.0f;
        return float3(
            r * cos(phi),
            r * sin(phi),
            sqrt(max(1.0f - r * r, 0.0f))
        );
    }
}

float raytraceSphere(float3 origin, float3 dir, float3 center, float radius)
{
    float3 o = origin - center;
    float B = 2.0f * dot(o, dir);
    float C = dot(o, o) - radius * radius;
    float mantissa = B * B - 4.0f * C;
    if (mantissa < 0.0f)
        return 0.0f;

    return (-B - sqrt(mantissa)) * 0.5f;
}

// Schlick Fresnel approximation for specular reflections.
[Differentiable]
float SchlickFresnel(float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m; // pow(m,5)
}

[Differentiable]
float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return a2 / (float.getPi() * t * t);
}

[Differentiable]
float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

// Simplified Disney BRDF
struct DisneyBRDF {
    float3 baseColor;
    float3 normal;
    float roughness;
    float metallic;
    float specular;

    float3 eval(float3 wiLocal, float3 woLocal)
    {
        float NdotV = dot(normal, wiLocal);
        float NdotL = dot(normal, woLocal);
        if (NdotL < 0 || NdotV < 0)
            return float3(0.0f);

        float3 H = normalize(woLocal + wiLocal);
        float NdotH = dot(normal, H);
        float LdotH = dot(woLocal, H);

        float3 Cdlin = baseColor;
        float Cdlum = .3f * Cdlin[0] + .6f * Cdlin[1] + .1f * Cdlin[2]; // luminance approx.

        float3 Ctint = Cdlum > 0 ? Cdlin / Cdlum : float3(1); // normalize lum. to isolate hue+sat
        float3 Cspec0 = lerp(float3(specular * 0.08f), Cdlin, metallic);

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
        // and lerp in diffuse retro-reflection based on roughness
        float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
        float Fd90 = 0.5f + 2 * LdotH * LdotH * roughness;
        float Fd = lerp(1.f, Fd90, FL) * lerp(1.f, Fd90, FV);

        // specular
        float a = roughness * roughness;
        float Ds = GTR2(NdotH, a);
        float FH = SchlickFresnel(LdotH);
        float3 Fs = lerp(Cspec0, float3(1), FH);
        float Gs;
        Gs = smithG_GGX(NdotL, a);
        Gs *= smithG_GGX(NdotV, a);

        return NdotL * (((1.0f / float.getPi()) * Fd * Cdlin) * (1 - metallic) + Gs * Fs * Ds);
    }
}
