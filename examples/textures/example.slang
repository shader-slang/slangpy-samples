// SPDX-License-Identifier: Apache-2.0

public module;

// Add an amount to a given pixel
void brighten(float4 amount, inout float4 pixel)
{
    pixel += amount;
}
