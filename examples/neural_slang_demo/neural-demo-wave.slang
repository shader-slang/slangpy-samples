// SPDX-License-Identifier: Apache-2.0
import neural;

static const int SubgroupSize = 32;
static const int SubgroupCount = 1;

// SharedMemorySize for the 3 WaveTangledVector layers: 4->32, 32->32, 32->3
typealias ShMemSize = SharedMemorySize<float, TargetEnum.SPIR_V, ExecutionMode.Training, SubgroupSize, SubgroupCount>;
typealias ShMemSizeForNetwork = ShMemSize.OfLayer3<4, 32, 32, 3>;

export struct InputVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 4, SubgroupSize>;
export struct HiddenVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 32, SubgroupSize>;
export struct OutputVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 3, SubgroupSize>;

export float3 outputToFloat3(WaveTangledVector<float, ShMemSizeForNetwork, 3, SubgroupSize> v) { return float3(v[0], v[1], v[2]); }
