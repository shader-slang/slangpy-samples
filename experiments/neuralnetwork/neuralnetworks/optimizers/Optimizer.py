# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np

from slangpy.types import NDBuffer
from ..basetypes import Real

from slangpy import InstanceList, Module, Tensor, CommandEncoder, pack
from slangpy.core.function import FunctionNode

from typing import Any, Optional

class OptimizerPool:
    def __init__(self, module: Module, optim_type_name: str):
        optim_type = module.find_struct(optim_type_name)
        if optim_type is None:
            raise ValueError(f"Could not find optimizer type '{optim_type_name}' in slang module '{module.name}'. "
                                "This could be due to a missing import or a type error. Make sure "
                                "this is a valid type in the module, e.g. by pasting in the type above "
                                "and checking for compile errors")

        batch_type = module.find_struct(f"{optim_type_name}::Batch")
        if batch_type is None:
            raise ValueError(f"Could not find optimizer batch type '{optim_type_name}::State' in slang module "
                                f"'{module.name}'. Make sure the type {optim_type_name} implements IOptimizer")

        state_type = module.find_struct(f"{optim_type_name}::State")
        if state_type is None:
            raise ValueError(f"Could not find optimizer state type '{optim_type_name}::State' in slang module "
                                f"'{module.name}'. Make sure the type {optim_type_name} implements IOptimizer")

        step_func = module.find_function_in_struct(optim_type, "step")
        if step_func is None:
            raise ValueError(f"Could not find method '{optim_type_name}::step()' in slang module '{module.name}'. "
                                f"Make sure the type {optim_type_name} implements IOptimizer")

        batch_step_func = module.find_function_in_struct(optim_type, "batch_step")
        if batch_step_func is None:
            raise ValueError(f"Could not find method '{optim_type_name}::batch_step()' in slang module '{module.name}'. "
                                f"Make sure the type {optim_type_name} implements IOptimizer")

        self.optim_type = optim_type
        self.state_type = state_type
        self.batch_type = batch_type
        self.step_func = step_func
        self.batch_step_func = batch_step_func
        self.params: list[Tensor] = []
        self.mapping = np.ndarray((0,2), dtype=np.int32)
        self.batches: list[dict[str,Any]] = []

    def finalise(self):
        """
        Finalizes the optimizer pool, preparing it for use.
        This is called after all parameters have been added.
        """
        # Convert the mapping to a packed array
        self.mapping_buffer = NDBuffer(self.optim_type.module.device, dtype="int2", element_count=self.mapping.shape[0])
        self.mapping_buffer.copy_from_numpy(self.mapping)

        # Create the packed batch data
        self.batches_packed = pack(self.optim_type.module, self.batches)

        self.batch_step_func = self.optim_type.module.find_function_in_struct(self.optim_type, f"batch_step<{len(self.batches)}>")

    def add_parameter(self, param: Tensor):
        """
        Adds a parameter to the optimizer pool.
        """

        param_idx = len(self.params)
        self.params.append(param)

        self.batches.append(InstanceList(self.batch_type,{
            "params": param.detach().storage,
            "grads": param.grad.storage,
            "states": self.state_type(param).storage,
        }))

        # Append to the mapping array 1 entry for each element of the tensor,
        # where the entry is [param_idx, element_idx]
        # Efficiently append all [param_idx, i] pairs for i in range(param.element_count)
        new_mapping = np.column_stack((
            np.full(param.element_count, param_idx, dtype=np.int32),
            np.arange(param.element_count, dtype=np.int32)
        ))
        self.mapping = np.vstack([self.mapping, new_mapping])

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
        self.packed_primals = [pack(module,param) for param in parameters]
        self.packed_grads = [pack(module,param.grad) for param in parameters]
        self.states = []
        self.pools: dict[str, OptimizerPool] = {}
        self.step_funcs: list[FunctionNode] = []

        for i, param in enumerate(parameters):
            dtype = Real.from_slangtype(param.dtype)
            if dtype is None:
                raise ValueError(f"Unsupported element type '{param.dtype.full_name}' "
                                 f"of parameter {i}: Must be half, float or double")

            type_name = self.get_type_name(dtype)

            pool = self._get_or_create_optimizer_pool(module, type_name)
            pool.add_parameter(param)

            optim_type = module.find_struct(type_name)
            if optim_type is None:
                raise ValueError(f"Could not find optimizer type '{type_name}' in slang module '{module.name}'. "
                                 "This could be due to a missing import or a type error. Make sure "
                                 "this is a valid type in the module, e.g. by pasting in the type above "
                                 "and checking for compile errors")

            state_type = module.find_struct(f"{type_name}::State")
            if state_type is None:
                raise ValueError(f"Could not find optimizer state type '{type_name}::State' in slang module "
                                 f"'{module.name}'. Make sure the type {type_name} implements IOptimizer<{dtype.slang()}>")

            step_func = module.find_function_in_struct(optim_type, "step")
            if step_func is None:
                raise ValueError(f"Could not find method '{type_name}::step()' in slang module '{module.name}'. "
                                 f"Make sure the type {type_name} implements IOptimizer<{dtype.slang()}>")

            self.states.append(pack(module,state_type(param)))
            self.step_funcs.append(step_func)

        for pool in self.pools.values():
            pool.finalise()

        pass

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
        #for primal, grad, state, step_func in zip(self.packed_primals, self.packed_grads, self.states, self.step_funcs):
        #    if cmd is None:
        #        step_func(this, state, primal, grad)
        #    else:
        #        step_func.append_to(cmd, this, state, primal, grad)
        #for pool in self.pools.values():
        #    for batch in pool.batches:
        #        if cmd is None:
        #            pool.step_func(this, batch["states"], batch["params"], batch["grads"])
        #        else:
        #            pool.step_func.append_to(cmd, this, batch["states"], batch["params"], batch["grads"])
        for pool in self.pools.values():
            if cmd is None:
                pool.batch_step_func(this, pool.batches_packed, pool.mapping_buffer)
            else:
                pool.batch_step_func.append_to(cmd, this, pool.batches_packed, pool.mapping_buffer)

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
            raise RuntimeError("Optimizer is uninitialized. Make sure to "
                               "call .initialize() before using the optimizer")
