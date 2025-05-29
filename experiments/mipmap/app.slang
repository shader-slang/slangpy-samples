// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import slangpy;

// Helper function to turn pixel coordinates into normalized UVs.
float2 pixel_to_uv(int2 pixel, int2 resolution)
{
    return (float2(pixel) + 0.5f) / float2(resolution);
}

void blit(
    int2 pixel,
    int2 size,
    int2 offset,
    Tensor<float3, 2> input,
    RWTexture2D<float4> output,
)
{
    uint2 output_dimensions;
    output.GetDimensions(output_dimensions.x, output_dimensions.y);

    float2 uv = pixel_to_uv(pixel, size);

    uint[2] shape = input.shape;

    float3 col = input.get({
        uint(shape[0]*uv.y),
        uint(shape[1]*uv.x)
    });

    output[offset+pixel] = float4(col, 1.0f);
}
