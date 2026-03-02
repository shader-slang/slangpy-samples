// SPDX-License-Identifier: Apache-2.0
import neural;

export struct InputVec : IVector<float> = InlineVector<float, 4>;
export struct HiddenVec : IVector<float> = InlineVector<float, 32>;
export struct OutputVec : IVector<float> = InlineVector<float, 3>;

export float3 outputToFloat3(InlineVector<float, 3> v) { return float3(v[0], v[1], v[2]); }
