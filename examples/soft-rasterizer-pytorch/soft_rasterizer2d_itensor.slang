// SPDX-License-Identifier: Apache-2.0

import "slangpy";

struct Camera
{
    float2 o;         // world origin
    float2 scale;     // world scale
    float2 frameDim;  // image resolution (W, H)

    [NoDiffThis]
    float2 screenToWorld(float2 uv)
    {
        float2 xy = uv / frameDim;
        float2 ndc = xy * 2.0f - 1.0f;
        return ndc * scale + o;
    }
};

[Differentiable]
[PreferRecompute]
float distanceToVertex(float2 xy, float2 v)
{
    return length(xy - v);
}

[Differentiable]
[PreferRecompute]
float distanceToEdge(float2 u, float2 v, float2 pt)
{
    float2 e = v - u;
    float2 n = float2(-e.y, e.x);
    float2 d = pt - u;
    float n_dot_d = dot(n, d);

    if (dot(pt - u, v - u) < 0.f)
        return distanceToVertex(pt, u) * sign(n_dot_d);
    else if (dot(pt - v, u - v) < 0.f)
        return distanceToVertex(pt, v) * sign(n_dot_d);
    else
        return n_dot_d / length(n);
}

[Differentiable]
[PreferRecompute]
float distanceToTriangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    float d1 = distanceToEdge(v2, v1, xy);
    float d2 = distanceToEdge(v3, v2, xy);
    float d3 = distanceToEdge(v1, v3, xy);

    float sd1 = sign(d1);
    float sd2 = sign(d2);
    float sd3 = sign(d3);

    if (sd1 > 0.f && sd2 > 0.f)
        return distanceToVertex(xy, v2);
    else if (sd1 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v1);
    else if (sd2 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v3);
    else
        return max(max(d1, d2), d3);
}

[Differentiable]
[PreferRecompute]
float sigmoid(float x, float sigma)
{
    return 1.0 / (1.0 + exp(-x / sigma));
}

[Differentiable]
[PreferRecompute]
float softTriangle(float2 xy, float2 v1, float2 v2, float2 v3, float sigma)
{
    float d = distanceToTriangle(xy, v1, v2, v3);
    return sigmoid(-d, sigma);
}

// Per-pixel soft rasterizer (SlangPy style):
// - vertices: ITensor<float,2> with shape [3,2]
// - color:    ITensor<float,1> with shape [3]
// - grid_cell: pixel coordinate supplied by SlangPy via spy.call_id()
[Differentiable]
float3 rasterize_pixel(
    uniform Camera camera,
    ITensor<float, 2> vertices,
    ITensor<float, 1> color,
    float sigma,
    uint2 grid_cell)
{
    float2 v1 = float2(vertices[0, 0], vertices[0, 1]);
    float2 v2 = float2(vertices[1, 0], vertices[1, 1]);
    float2 v3 = float2(vertices[2, 0], vertices[2, 1]);
    float3 c  = float3(color[0], color[1], color[2]);
    float3 background = float3(1.0);

    float2 pixel = float2(grid_cell) + 0.5;
    float2 ndc = (pixel / camera.frameDim) * 2.0f - 1.0f;
    float2 world_sample = ndc * camera.scale + camera.o;

    float hit = softTriangle(world_sample, v1, v2, v3, sigma);
    return hit * c + (1.0 - hit) * background;
}
