// SPDX-License-Identifier: Apache-2.0

import "slangpy";

struct Pixel
{
    float r;
    float g;
    float b;
};

// Add 2 pixels together
Pixel add(Pixel a, Pixel b)
{
    Pixel result;
    result.r = a.r + b.r;
    result.g = a.g + b.g;
    result.b = a.b + b.b;
    return result;
}
