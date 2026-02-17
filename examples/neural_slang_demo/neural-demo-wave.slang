// SPDX-License-Identifier: Apache-2.0
import neural;

static const int SubgroupSize = 32;
static const int SubgroupCount = 1;

// SharedMemoryPool for the 2 WaveTangledVector layers: 32->32 and 32->3
typealias ShMemSize = SharedMemorySize<float, TargetEnum.SPIR_V, ExecutionMode.Training, SubgroupSize, SubgroupCount>;
typealias ShMemSizeForNetwork = ShMemSize.OfLayer2<32, 32, 3>;
typealias ShMemPool = SharedMemoryPool<ShMemSizeForNetwork>;

export struct HiddenVec : IVector<float> = WaveTangledVector<float, ShMemPool, 32, SubgroupSize>;
