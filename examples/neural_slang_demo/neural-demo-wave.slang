// SPDX-License-Identifier: Apache-2.0
import neural;

static const int SubgroupSize = 32;
static const int SubgroupCount = 1;

// SharedMemoryPool for the 3 WaveTangledVector layers: 4->32, 32->32, 32->3
typealias ShMemSize = SharedMemorySize<float, TargetEnum.SPIR_V, ExecutionMode.Training, SubgroupSize, SubgroupCount>;
typealias ShMemSizeForNetwork = ShMemSize.OfLayer3<4, 32, 32, 3>;
typealias ShMemPool = SharedMemoryPool<ShMemSizeForNetwork>;

export struct InputVec : IVector<float> = WaveTangledVector<float, ShMemPool, 4, SubgroupSize>;
export struct HiddenVec : IVector<float> = WaveTangledVector<float, ShMemPool, 32, SubgroupSize>;
export struct OutputVec : IVector<float> = WaveTangledVector<float, ShMemPool, 3, SubgroupSize>;
