# SPDX-License-Identifier: Apache-2.0

import numpy as np

from slangpy.types import NDBuffer
from ..basetypes import Real

from slangpy import Buffer, InstanceList, Module, Tensor, CommandEncoder, pack
from slangpy.core.function import FunctionNode

from typing import Any, Optional


class OptimizerPool:
    def __init__(self, module: Module, optim_type_name: str):
        optim_type = module.find_struct(optim_type_name)
        if optim_type is None:
            raise ValueError(
                f"Could not find optimizer type '{optim_type_name}' in slang module '{module.name}'. "
                "This could be due to a missing import or a type error. Make sure "
                "this is a valid type in the module, e.g. by pasting in the type above "
                "and checking for compile errors"
            )

        batch_type = module.find_struct(f"{optim_type_name}::Batch")
        if batch_type is None:
            raise ValueError(
                f"Could not find optimizer batch type '{optim_type_name}::State' in slang module "
                f"'{module.name}'. Make sure the type {optim_type_name} implements IOptimizer"
            )

        state_type = module.find_struct(f"{optim_type_name}::State")
        if state_type is None:
            raise ValueError(
                f"Could not find optimizer state type '{optim_type_name}::State' in slang module "
                f"'{module.name}'. Make sure the type {optim_type_name} implements IOptimizer"
            )

        step_func = module.find_function_in_struct(optim_type, "step")
        if step_func is None:
            raise ValueError(
                f"Could not find method '{optim_type_name}::step()' in slang module '{module.name}'. "
                f"Make sure the type {optim_type_name} implements IOptimizer"
            )

        batch_step_func = module.find_function_in_struct(optim_type, "batch_step")
        if batch_step_func is None:
            raise ValueError(
                f"Could not find method '{optim_type_name}::batch_step()' in slang module '{module.name}'. "
                f"Make sure the type {optim_type_name} implements IOptimizer"
            )

        self.optim_type = optim_type
        self.state_type = state_type
        self.batch_type = batch_type
        self.step_func = step_func
        self.batch_step_func = batch_step_func
        self.params: list[Tensor] = []
        self.states: list[NDBuffer] = []
        self.mapping = np.ndarray((0, 2), dtype=np.int32)
        self.batched_params: list[dict[str, Any]] = []
        self.unbatched_params: list[dict[str, Any]] = []

    def finalise(self):
        """
        Finalizes the optimizer pool, preparing it for use.
        This is called after all parameters have been added.
        """
        if len(self.batched_params) > 0:
            # Convert the mapping to a packed array
            self.mapping_buffer = NDBuffer(
                self.optim_type.module.device, dtype="int2", element_count=self.mapping.shape[0]
            )
            self.mapping_buffer.copy_from_numpy(self.mapping)

            # Create the packed batch data
            self.batches_packed = pack(self.optim_type.module, self.batched_params)

            # Workaround for Slang reflection issue - explicitly specialize batch_step
            # for the number of batches we have, so that Slang can find the correct function.
            self.batch_step_func = self.optim_type.module.find_function_in_struct(
                self.optim_type, f"batch_step<{len(self.batched_params)}>"
            )
        else:
            self.mapping_buffer = None
            self.batches_packed = None

    def add_parameter(self, param: Tensor, existing_state: Optional[NDBuffer] = None):
        """
        Adds a parameter to the optimizer pool.
        """

        self.params.append(param)

        # If no existing state is provided, create a new state for the parameter
        if existing_state is None:
            state = self.state_type(param)
        else:
            assert existing_state.element_count == param.element_count
            state = existing_state
        self.states.append(state)

        if param.element_count >= 32 * 1024:
            # If parameter is large enough, it's suitable for a single dispatch
            self.unbatched_params.append(
                {
                    "params": param.detach(),
                    "grads": param.grad,
                    "states": state,
                }
            )
        else:
            # Small parameters are batched together into 1 mega dispatch with an LUT
            batch_idx = len(self.batched_params)
            self.batched_params.append(
                InstanceList(
                    self.batch_type,
                    {
                        "params": param.storage,
                        "grads": param.grad.storage,
                        "states": state.storage,
                    },
                )
            )

            # Append to the mapping array 1 entry for each element of the tensor,
            # where the entry is [param_idx, element_idx]
            new_mapping = np.column_stack(
                (
                    np.full(param.element_count, batch_idx, dtype=np.int32),
                    np.arange(param.element_count, dtype=np.int32),
                )
            )
            self.mapping = np.vstack([self.mapping, new_mapping])

    def prune(self, parameters_to_keep: set[Tensor]):
        """
        Prunes the optimizer pool, removing any parameters that are no longer needed.
        This is called when the optimizer is no longer used.
        """
        curr_params = self.params
        curr_states = self.states

        self.params = []
        self.states = []
        self.mapping = np.ndarray((0, 2), dtype=np.int32)
        self.batched_params = []
        self.unbatched_params = []

        for param, state in zip(curr_params, curr_states):
            if param in parameters_to_keep:
                self.add_parameter(param, existing_state=state)

        self.finalise()


class Optimizer:
    """
    This is the base class of all optimizers.

    Creating an optimizer is done in two phases: First, by calling the constructor
    of the optimizer (e.g. AdamOptimizer) and setting its parameters. This is light-weight
    and does not do much work yet.
    Second, by calling .initialize(module, parameters), passing in the slang module containing
    the required slang types and a list of network parameters to optimize. This may perform
    allocation and reflection work.

    .step() performs one optimization step and resets the network gradients.

    For implementers of new optimizers, the following methods should be overridden:
    - get_type_name(dtype) returning the name of a slang type implementing IOptimizer<dtype>
    - get_this(), returning a python type that may be passed to slang (e.g. a dict)
    """

    def __init__(self):
        super().__init__()
        self._initialized = False

    def initialize(self, module: Module, parameters: list[Tensor]):
        """
        Initializes the optimizer from a list of trainable parameters.

        The optimizer must be initialized before it can be used.

        module is a loaded slang module containing the required slang types.

        Parameter tensors don't all have to have the same precision, and it is allowed to use networks
        with e.g. mixed float and half precision parameters.
        """
        self._initialized = True
        self.parameters = parameters
        self.pools: dict[str, OptimizerPool] = {}

        for i, param in enumerate(parameters):
            dtype = Real.from_slangtype(param.dtype)
            if dtype is None:
                raise ValueError(
                    f"Unsupported element type '{param.dtype.full_name}' "
                    f"of parameter {i}: Must be half, float or double"
                )

            type_name = self.get_type_name(dtype)

            pool = self._get_or_create_optimizer_pool(module, type_name)
            pool.add_parameter(param)

        for pool in self.pools.values():
            pool.finalise()

    def _get_or_create_optimizer_pool(self, module: Module, optim_type_name: str) -> OptimizerPool:
        """
        Returns an existing optimizer pool for the given type, or creates a new one if it does not exist.
        """
        if optim_type_name not in self.pools:
            self.pools[optim_type_name] = OptimizerPool(module, optim_type_name)
        return self.pools[optim_type_name]

    def step(self, cmd: Optional[CommandEncoder] = None):
        """
        Performs one step of the optimizer and resets network gradients.

        If cmd is provided, the slang calls are appended to the given command buffer.
        """
        self.check_initialized()

        this = self.get_this()
        for pool in self.pools.values():
            if cmd is None:
                if pool.batches_packed is not None:
                    pool.batch_step_func(this, pool.batches_packed, pool.mapping_buffer)
                for param in pool.unbatched_params:
                    pool.step_func(this, param["states"], param["params"], param["grads"])
            else:
                if pool.batches_packed is not None:
                    pool.batch_step_func.append_to(
                        cmd, this, pool.batches_packed, pool.mapping_buffer
                    )
                for param in pool.unbatched_params:
                    pool.step_func.append_to(
                        cmd, this, param["states"], param["params"], param["grads"]
                    )

    def get_type_name(self, dtype: Real) -> str:
        """Returns the name of a slang type implementing IOptimizer<dtype>"""
        raise NotImplementedError()

    def get_this(self):
        """
        Returning a python type that may be passed to slang (e.g. a dict)

        Currently, this type has to be compatible with any optimizer precision.
        This may change in the future.
        """
        raise NotImplementedError()

    def check_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                "Optimizer is uninitialized. Make sure to "
                "call .initialize() before using the optimizer"
            )
