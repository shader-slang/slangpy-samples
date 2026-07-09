// SPDX-License-Identifier: Apache-2.0
// Link module: provides the concrete vector types for the extern declarations in
// neural-demo.slang.
import slang.neural;

export struct InputVec : IVector<float> = InlineVector<float, 4>;
export struct HiddenVec : IVector<float> = InlineVector<float, 32>;
export struct OutputVec : IVector<float> = InlineVector<float, 3>;
