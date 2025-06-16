# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pathlib
from slangpy import Device, Tensor
from .basetypes.Real import Real


def slang_include_paths() -> list[pathlib.Path]:
    return [pathlib.Path(__file__).parent / "slang"]


def merge_tensors(tensors: list[Tensor], alignment: int = 0) -> list[Tensor]:
    """
    Transforms list of input tensors into views of a larger merged tensor.

    This is helpful for reducing the number of dispatches the optimizer needs
    to do by merging a potentially large number of parameter buffers allocated
    by a model into a small number of combined tensors.

    Tensors are grouped by element type, and for each group, a
    1D tensor is allocated to hold the combined input tensors.
    The input tensors are then turned into views of this larger tensor.
    The original data is copied over, and the merged tensors are returned.

    For tensors that need to maintain a specific alignment (e.g. coopvec matrices),
    an alignment value (in bytes) can be specified.

    Note that the input tensors are modified in place to point at the merged tensor.
    After the call, the input tensors will be dense views, and e.g. striding tricks will
    not be maintained. Gradient tensors are also merged, and will be zeroed by this call.
    """
    tensors_by_dtype: dict[Real, list[Tensor]] = {
        Real.half: [],
        Real.float: [],
        Real.double: [],
    }

    for i, tensor in enumerate(tensors):
        dtype = Real.from_slangtype(tensor.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported element type '{tensor.dtype.full_name}' "
                                f"of tensor {i}: Must be half, float or double")

        tensors_by_dtype[dtype].append(tensor)

    result: list[Tensor] = []

    for dtype, dtype_tensor in tensors_by_dtype.items():
        if len(dtype_tensor) == 0:
            continue

        # Turn alignment in terms of bytes into alignment in terms of elements
        dtype_size = dtype.size()
        if alignment < dtype_size:
            element_alignment = 1
        else:
            element_alignment = alignment // dtype_size
        if (element_alignment * dtype_size % alignment) != 0:
            raise ValueError(f"Requested alignment of {alignment} can't be satisfied with {dtype}")

        # Compute aligned offsets for each tensor and total combined size
        offsets = []
        offset = 0
        for param_tensor in dtype_tensor:
            misalignment = offset % element_alignment
            if misalignment != 0:
                offset += element_alignment - misalignment

            offsets.append(offset)
            offset += param_tensor.element_count

        total_count = offset

        # Allocate merged tensor
        merged_params = Tensor.empty(dtype_tensor[0].device, (total_count, ), dtype_tensor[0].dtype)
        merged_params = merged_params.with_grads(zero=True)

        # Finally, slice merged tensor into input tensors
        for offset, param_tensor in zip(offsets, dtype_tensor):
            n = param_tensor.element_count

            param_slice = merged_params.view(param_tensor.shape, offset=offset)
            param_slice.copy_from_numpy(param_tensor.to_numpy())
            param_tensor.point_to(param_slice)

            if param_tensor.grad_out is not None:
                grad_slice = merged_params.grad_out.view(param_tensor.shape, offset=offset)
                param_tensor.grad_out.point_to(grad_slice)


        result.append(merged_params)

    return result


def coopvec_matrix_alignment(device: Device):
    """Returns the required alignment (in bytes) of coopvec matrices"""
    return device.coopvec_align_matrix_offset(1)


def coopvec_vector_alignment(device: Device):
    """Returns the required alignment (in bytes) of coopvec vector"""
    return device.coopvec_align_vector_offset(1)
