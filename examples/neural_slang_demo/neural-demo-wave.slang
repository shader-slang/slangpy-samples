// SPDX-License-Identifier: Apache-2.0
// Link module: provides the concrete vector types for the extern declarations in
// neural-demo.slang.
import slang.neural;

// slangpy generates [numthreads(32, 1, 1)] kernels: one 32-wide subgroup per group.
static const int SubgroupSize = 32;
static const int SubgroupCount = 1;

// SharedMemorySize needs the compile target to size the pool; the Python driver
// defines NEURAL_DEMO_TARGET from the selected device type.
#ifndef NEURAL_DEMO_TARGET
#define NEURAL_DEMO_TARGET TargetEnum.SPIR_V
#endif

// Shared memory pool sized for training the 4 -> 32 -> 32 -> 3 network.
typealias ShMemSizeForNetwork = SharedMemorySize<
    float, NEURAL_DEMO_TARGET, ExecutionMode.Training, SubgroupSize, SubgroupCount,
    4, 32, 32, 3>;

export struct InputVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 4, SubgroupSize, SubgroupCount>;
export struct HiddenVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 32, SubgroupSize, SubgroupCount>;
export struct OutputVec : IVector<float> = WaveTangledVector<float, ShMemSizeForNetwork, 3, SubgroupSize, SubgroupCount>;
