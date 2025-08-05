// SPDX-License-Identifier: Apache-2.0

import "slangpy";

/// Simple Camera struct to convert between screen space and world space
/// coordinates, so that vertex positions can be specified in world space.
struct Camera
{
    /// World origin.
    float2 origin;

    /// World scale.
    float2 scale;

    /// Frame dimensions (i.e. image resolution).
    float2 frameDim;

    /// Convert from screen coordinates
    ///   [(0, 0), (W, H)]
    /// to world coordinates:
    ///   [(origin.x - scale.x, origin.y - scale.y), (origin.x + scale.x, origin.y + scale.y)]
    ///
    /// @param uv Screen coordinates
    /// @return World coordinates
    [Differentiable]
    [NoDiffThis]
    float2 screenToWorld(float2 uv)
    {
        float2 xy = uv / frameDim;
        float2 ndc = xy * 2.0f - 1.0f;
        return ndc * scale + origin;
    }
};

/// Taking the reciprocal of a value can lead to that value going infinite,
/// resulting in invalid gradients. This function uses an experimentally-
/// determined value to avoid this.
///
/// @param x input value
/// @return 1 / (x+epsilon)
[Differentiable]
float safe_reciprocal(float x)
{
    const float EPSILON = 1e-8; // experimentally-determined value
    return 1.0 / (x + EPSILON);
}

/// Compute the distance from a point to a vertex
///
/// @param xy2 point
/// @param v vertex
/// @return distance
[Differentiable]
float distanceToVertex(float2 xy2, float2 v)
{
    return length(xy2 - v);
}

/// Compute the signed distance from a point to an edge
///
/// @param u vertex 1 of edge
/// @param v vertex 2 of edge
/// @param pt point
/// @return distance
[Differentiable]
float distanceToEdge(float2 u, float2 v, float2 pt)
{
    float2 e = v - u;
    float2 n = float2(-e.y, e.x);
    float2 d = pt - u;
    float n_dot_d = dot(n, d);

    // Compute the signed distance from a point to an edge
    if (dot(pt - u, v - u) < 0.f)
        return distanceToVertex(pt, u) * (sign(n_dot_d)); // u is the closest vertex
    else if (dot(pt - v, u - v) < 0.f)
        return distanceToVertex(pt, v) * (sign(n_dot_d)); // v is the closest vertex
    else
    {
        return n_dot_d * safe_reciprocal(length(n)); // closest edge
    }
}

/// Compute the signed distance from a point to a triangle
///
/// @param xy point
/// @param v1 vertex 1 of triangle
/// @param v2 vertex 2 of triangle
/// @param v3 vertex 3 of triangle
/// @return distance
[Differentiable]
float distanceToTriangle(float2 xy, float2 v1, float2 v2, float2 v3)
{
    // Minimum distance to the edge of the triangle
    float d1 = distanceToEdge(v2, v1, xy);
    float d2 = distanceToEdge(v3, v2, xy);
    float d3 = distanceToEdge(v1, v3, xy);

    int sd1 = sign(d1);
    int sd2 = sign(d2);
    int sd3 = sign(d3);

    if (sd1 > 0.f && sd2 > 0.f)
        return distanceToVertex(xy, v2); // v2 is the closest vertex
    else if (sd1 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v1); // v1 is the closest vertex
    else if (sd2 > 0.f && sd3 > 0.f)
        return distanceToVertex(xy, v3); // v3 is the closest vertex
    else
        return max(max(d1, d2), d3);
}


/// Compute leaky ReLU
///
/// @param x input value
/// @param negative_slope
/// @return x, if x>=0, otherwise returns negative_slope * x
[Differentiable]
float leakyReLU(float x, no_diff float negative_slope = 1e-2)
{
    return max(0, x) + negative_slope * min(0, x);
}


/// Compute by how much a given point is inside a soft triangle.
/// The result is scaled so that the values between [0,1], which will be
/// visible when displayed, show a soft edge of suitable thickness.
///
/// @param xy point
/// @param v1 vertex 1 of triangle
/// @param v2 vertex 2 of triangle
/// @param v3 vertex 3 of triangle
/// @param edge_scale scaling factor for soft edge thickness (higher=thinner)
/// @param leakyrelu_slope negative slope value to use with Leaky ReLU
/// @return 1 - the scaled distance from the edge
[Differentiable]
float softTriangle(float2 xy, float2 v1, float2 v2, float2 v3, no_diff float edge_scale, no_diff float leakyrelu_slope)
{
    float d = distanceToTriangle(xy, v1, v2, v3);
    return leakyReLU(1 - edge_scale*d);
}


/// Rasterize function called from python, called for each pixel in the window.
/// The soft rasterizer instead computes a probability value for a pixel being
/// within the triangle, rather than the simple boolean value of the
/// fwd-rasterizer. These functions are differentiable, allowing returning
/// gradient values for training.
///
/// @param camera Camera struct
/// @param vertices 2D vertices for the triangle
/// @param pixelCoord Pixel coordinates to evaluate
/// @return Green if inside triangle, fading to Block based on distance otherwise
[Differentiable]
float4 rasterize(Camera camera, float2 vertices[3], int2 pixelCoord)
{
    // Adjust pixel coordinate to have 0,0 at the bottom left for ease of use.
    float2 fragCoord = float2(pixelCoord);
    fragCoord.y = camera.frameDim.y - fragCoord.y;

    // Three vertices of the triangle, passed in from python.
    float2 v1 = vertices[0];
    float2 v2 = vertices[1];
    float2 v3 = vertices[2];

    // Convert the screen space fragCoord to world space.
    float2 screenSample = fragCoord.xy + 0.5;
    float2 worldSample = camera.screenToWorld(screenSample);

    // Negative slope for LeakyReLU
    float leakyrelu_slope = 0.2;

    // Soft edge scale factor
    float edge_scale = 20;

    // Green triangle color.
    float4 triangleColor = float4(0.0, 1.0, 0.0, 1.0);

    // Compute hit probability for the pixel being inside the triangle bounds.
    float hit = softTriangle(worldSample, v1, v2, v3, edge_scale, leakyrelu_slope);
    float4 outColor = hit * triangleColor;
    return outColor;
}


/// Calculate the difference between a rendered pixel and a reference pixel at
/// a given set of coordinates, when rasterizing a soft triangle with given
/// vertices.
///
/// @param camera Camera struct for the reference Tensor
/// @param vertices Vertices of the triangle
/// @param pixelCoord Pixel coordinates to evaluate
/// @param reference Reference value at those pixel coordinates
[Differentiable]
float4 loss(Camera camera, float2 vertices[3], int2 pixelCoord, no_diff float4 reference)
{
    float4 color = rasterize(camera, vertices, pixelCoord);
    float4 error = color - reference;
    return error * error; // Squared error
}


/// Generate the gradients at a given set of pixel coordinates.
///
/// @param camera Camera struct for the reference Tensor
/// @param primal Tensor containing the primal vertex values
/// @param grad Tensor containing the gradients for the vertex values
/// @param pixelCoord Pixel coordinates to evaluate
/// @param reference Reference value at those pixel coordinates
void generate_gradients(Camera camera, in Tensor<float2, 1> primal, AtomicTensor<float2, 1> grad, in int2 pixelCoord, in float4 reference)
{
    // Wrap the vertices in a diffPair so we can use autodiff to get the
    // partial derivatives and calculate gradients
    float2 vertices[3] = { primal.get({0}), primal.get({1}), primal.get({2}) };
    var dVertices = diffPair(vertices);

    bwd_diff(loss)(camera, dVertices, pixelCoord, reference, 1.0);

    // Accumulate (add) gradients for this point into the gradients tensor.
    grad.set({0}, dVertices.d[0]);
    grad.set({1}, dVertices.d[1]);
    grad.set({2}, dVertices.d[2]);
}


/// Optimize the vertex values using gradient descent. Note that because we've
/// added the gradients unscaled in our generate_gradient function, we will
/// scale down grad by the overall number of samples during the optimize step.
///
/// @param primal Primal vertex value
/// @param grad Gradients for this vertex
/// @param learning_rate Learning rate
/// @param gradient_count Gradient count (scaling factor)
void optimize(inout float2 primal, inout float2 grad, in float learning_rate, in float gradient_count)
{
    // Primal moves in the opposite direction of the gradients
    primal -= learning_rate * (grad / gradient_count);
    // Clear the gradients Tensor in preparation for next frame
    grad = float2(0);
}


