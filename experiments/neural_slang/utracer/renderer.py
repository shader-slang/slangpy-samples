# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional, Union, cast

import numpy as np
from PIL import Image
import slangpy as spy
from slangpy import (
    AccelerationStructureBuildDesc,
    AccelerationStructureBuildFlags,
    AccelerationStructureBuildInputTriangles,
    AccelerationStructureBuildMode,
    AccelerationStructureGeometryFlags,
    Bitmap,
    Buffer,
    BufferOffsetPair,
    BufferUsage,
    Format,
    IndexFormat,
    Module,
    Tensor,
    float3,
    int2,
    uint2,
    uint3,
)
from slangpy.math import matrix_from_quat, normalize, quat_from_look_at, radians, tan
import slangpy.bindings as spybind
import slangpy.reflection as spyref

if __package__:
    from .checkpoint import BakedCheckpoint, load_baked_checkpoint
    from .common import VectorBackend, require_wave_safe_work_size
else:
    from checkpoint import BakedCheckpoint, load_baked_checkpoint
    from common import VectorBackend, require_wave_safe_work_size


class Camera:
    def __init__(
        self,
        device: spy.Device,
        width: int = 1024,
        height: int = 1024,
        fov_degrees: float = 45.0,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fov_degrees = fov_degrees
        self.pos = float3(1.8, 0.7, 0.4)
        self.rot = quat_from_look_at(-normalize(self.pos), float3(0, 1, 0))
        self.image_u = float3(1, 0, 0)
        self.image_v = float3(0, 1, 0)
        self.image_w = float3(0, 0, -1)
        self._output = self._create_texture()
        self._accumulator = self._create_texture()
        self.recompute()

    def recompute(self) -> None:
        aspect = float(self.width) / float(self.height)
        rotation = matrix_from_quat(self.rot)
        fov = radians(self.fov_degrees)
        self.image_u = rotation.get_col(0) * tan(fov * 0.5) * aspect
        self.image_v = rotation.get_col(1) * tan(fov * 0.5)
        self.image_w = -rotation.get_col(2)

    def get_uniforms(self) -> dict[str, object]:
        self.recompute()
        return {
            "position": self.pos,
            "imageU": self.image_u,
            "imageV": self.image_v,
            "imageW": self.image_w,
            "dimensions": int2(self.width, self.height),
        }

    def _create_texture(self) -> spy.Texture:
        return self.device.create_texture(
            width=self.width,
            height=self.height,
            format=Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        )

    def reset_accumulator(self, module: Module) -> None:
        module.accumulator_reset(self._accumulator)

    @property
    def output(self) -> spy.Texture:
        return self._output

    @property
    def accumulator(self) -> spy.Texture:
        return self._accumulator


class CameraMarshall(spybind.Marshall):
    def __init__(self, layout: spyref.SlangProgramLayout):
        super().__init__(layout)
        camera_type = layout.find_type_by_name("Camera")
        assert camera_type is not None
        self.slang_type = camera_type

    def get_shape(self, value: Camera):
        return spybind.Shape(value.height, value.width)

    def resolve_type(self, context: spybind.BindContext, bound_type: spyref.SlangType):
        return bound_type

    def resolve_dimensionality(
        self,
        context: spybind.BindContext,
        binding: spybind.BoundVariable,
        vector_target_type: spyref.SlangType,
    ):
        return 2 if vector_target_type.full_name == "RaySampler" else 0

    def create_calldata(
        self,
        context: spybind.BindContext,
        binding: spybind.BoundVariable,
        data: Camera,
    ):
        return data.get_uniforms()

    def gen_calldata(
        self,
        code: spybind.CodeGenBlock,
        context: spybind.BindContext,
        binding: spybind.BoundVariable,
    ):
        binding.gen_calldata_type_name(code, self.slang_type.full_name)


spybind.PYTHON_TYPES[Camera] = lambda layout, value: CameraMarshall(layout)


class MicroScene:
    """Single-mesh, single-neural-material scene."""

    def __init__(
        self,
        device: spy.Device,
        checkpoint: Union[str, PathLike[str]],
        vector_backend: VectorBackend = "inline",
        samples_per_pixel: int = 1,
    ) -> None:
        self.device = device
        self.vector_backend = vector_backend
        self.samples_per_pixel = samples_per_pixel
        self.module = Module(device.load_module("utracer"))
        self.checkpoint: BakedCheckpoint = load_baked_checkpoint(device, checkpoint, vector_backend)
        self.sample = 0

        self._vertices_np: Optional[np.ndarray] = None
        self._indices_np: Optional[np.ndarray] = None
        self._vertex_tensor: Optional[Tensor] = None
        self._index_tensor: Optional[Tensor] = None
        self._blas: Optional[spy.AccelerationStructure] = None
        self._tlas: Optional[spy.AccelerationStructure] = None
        self._blas_scratch: Optional[Buffer] = None
        self._tlas_scratch: Optional[Buffer] = None
        self._dirty = True

        self._sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            address_u=spy.TextureAddressingMode.wrap,
            address_v=spy.TextureAddressingMode.wrap,
        )
        self._black_texture = self._create_texture_from_numpy(np.zeros((1, 1, 4), dtype=np.float32))
        self._environment_texture: spy.Texture = self._black_texture
        self._environment_scale = float3(0.0)
        self._environment_valid = False

    def _create_texture_from_numpy(self, data: np.ndarray) -> spy.Texture:
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 3 or data.shape[2] != 4:
            raise ValueError(f"Expected HxWx4 texture, got {data.shape}")
        height, width = data.shape[:2]
        texture = self.device.create_texture(
            width=width,
            height=height,
            format=Format.rgba32_float,
            usage=spy.TextureUsage.shader_resource,
        )
        texture.copy_from_numpy(data)
        return texture

    def load_mesh(
        self,
        path: Union[str, PathLike[str]],
        rescale_to: Optional[float] = None,
    ) -> None:
        import trimesh

        loaded = trimesh.load(str(path), process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            if not meshes:
                raise ValueError(f"No meshes found in {path}")
            mesh = cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))
        else:
            mesh = cast(trimesh.Trimesh, loaded)

        if rescale_to is not None:
            maximum_extent = float(max(mesh.bounding_box.extents))
            if maximum_extent > 0.0:
                mesh.apply_scale(rescale_to / maximum_extent)
            mesh.apply_translation(-mesh.centroid)

        positions = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if normals.shape != positions.shape:
            normals = np.zeros_like(positions)

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uvs = np.asarray(mesh.visual.uv, dtype=np.float32)[:, :2]
        else:
            uvs = np.zeros((len(positions), 2), dtype=np.float32)

        tangents = _compute_tangents(positions, normals, uvs, faces)
        vertices = np.zeros((len(positions), 11), dtype=np.float32)
        vertices[:, 0:3] = positions
        vertices[:, 3:6] = normals
        vertices[:, 6:9] = tangents
        vertices[:, 9:11] = uvs
        self._vertices_np = vertices
        self._indices_np = faces.reshape(-1).astype(np.uint32)
        self._dirty = True

    def set_environment(
        self,
        path: Union[str, PathLike[str]],
        scale: float = 1.0,
    ) -> None:
        bitmap = Bitmap.load_from_file(str(path)).convert(
            pixel_format=Bitmap.PixelFormat.rgba,
            component_type=Bitmap.ComponentType.float32,
            srgb_gamma=False,
        )
        self._environment_texture = self._create_texture_from_numpy(np.asarray(bitmap))
        self._environment_scale = float3(scale)
        self._environment_valid = True

    def build(self) -> None:
        if not self._dirty:
            return
        if self._vertices_np is None or self._indices_np is None:
            raise ValueError("No mesh loaded")

        vertex_type = self.module.layout.require_type_by_name("Vertex")
        self._vertex_tensor = Tensor.empty(self.device, (len(self._vertices_np),), vertex_type)
        cursor = self._vertex_tensor.cursor()
        cursor.write_from_numpy(
            {
                "position": self._vertices_np[:, 0:3],
                "normal": self._vertices_np[:, 3:6],
                "tangent": self._vertices_np[:, 6:9],
                "uv": self._vertices_np[:, 9:11],
            },
            unchecked_copy=True,
        )
        cursor.apply()

        self._index_tensor = Tensor.empty(self.device, (len(self._indices_np),), "uint")
        index_cursor = self._index_tensor.cursor()
        index_cursor.write_from_numpy(self._indices_np, unchecked_copy=True)
        index_cursor.apply()
        self._build_blas()
        self._build_tlas()
        self._dirty = False

    def _build_blas(self) -> None:
        assert self._vertex_tensor is not None
        assert self._index_tensor is not None
        assert self._vertices_np is not None
        assert self._indices_np is not None

        triangles = AccelerationStructureBuildInputTriangles()
        triangles.vertex_buffers = [BufferOffsetPair(self._vertex_tensor.storage, 0)]
        triangles.vertex_count = len(self._vertices_np)
        triangles.vertex_stride = 11 * np.dtype(np.float32).itemsize
        triangles.vertex_format = Format.rgb32_float
        triangles.index_buffer = BufferOffsetPair(self._index_tensor.storage, 0)
        triangles.index_count = len(self._indices_np)
        triangles.index_format = IndexFormat.uint32
        triangles.flags = AccelerationStructureGeometryFlags.opaque

        description = AccelerationStructureBuildDesc()
        description.mode = AccelerationStructureBuildMode.build
        description.flags = AccelerationStructureBuildFlags.none
        description.inputs = [triangles]
        sizes = self.device.get_acceleration_structure_sizes(description)
        self._blas_scratch = self.device.create_buffer(
            sizes.scratch_size,
            usage=BufferUsage.unordered_access,
            label="utracer_blas_scratch",
        )
        self._blas = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size,
            label="utracer_blas",
        )
        command = self.device.create_command_encoder()
        command.build_acceleration_structure(
            desc=description,
            dst=self._blas,
            src=None,
            scratch_buffer=self._blas_scratch,
        )
        self.device.submit_command_buffer(command.finish())

    def _build_tlas(self) -> None:
        assert self._blas is not None
        instances = self.device.create_acceleration_structure_instance_list(1)
        instances.write(
            0,
            {
                "transform": spy.float3x4.identity(),
                "instance_id": 0,
                "instance_mask": 0xFF,
                "instance_contribution_to_hit_group_index": 0,
                "flags": spy.AccelerationStructureInstanceFlags.none,
                "acceleration_structure": self._blas.handle,
            },
        )
        description = AccelerationStructureBuildDesc()
        description.mode = AccelerationStructureBuildMode.build
        description.flags = AccelerationStructureBuildFlags.none
        description.inputs = [instances.build_input_instances()]
        sizes = self.device.get_acceleration_structure_sizes(description)
        self._tlas_scratch = self.device.create_buffer(
            sizes.scratch_size,
            usage=BufferUsage.unordered_access,
            label="utracer_tlas_scratch",
        )
        self._tlas = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size,
            label="utracer_tlas",
        )
        command = self.device.create_command_encoder()
        command.build_acceleration_structure(
            desc=description,
            dst=self._tlas,
            src=None,
            scratch_buffer=self._tlas_scratch,
        )
        self.device.submit_command_buffer(command.finish())

    def get_uniforms(self) -> dict[str, object]:
        assert self._vertex_tensor is not None
        assert self._index_tensor is not None
        return {
            "_type": "Scene",
            "tlas": self._tlas,
            "vertices": self._vertex_tensor.storage,
            "indices": self._index_tensor.storage,
            "material": {"latentTexture": self.checkpoint.latent_texture},
            "sampler": self._sampler,
            "environment": {
                "texture": self._environment_texture,
                "scale": self._environment_scale,
                "valid": self._environment_valid,
            },
        }

    def render(self, camera: Camera) -> spy.Texture:
        require_wave_safe_work_size(
            self.vector_backend,
            camera.width * camera.height,
            "render target",
        )
        self.build()
        command = self.device.create_command_encoder()
        self.module.render.set(
            {
                "gScene": self.get_uniforms(),
                "gNeuralParameters": self.checkpoint.parameter_address,
            }
        ).call(
            camera,
            camera.output,
            samplesPerPixel=self.samples_per_pixel,
            iteration=self.sample,
            _append_to=command,
        )
        self.module.accumulator_update(camera.output, camera.accumulator, _append_to=command)
        self.module.accumulator_output(camera.accumulator, camera.output, _append_to=command)
        self.sample += 1
        self.device.submit_command_buffer(command.finish())
        return camera.output


class Viewer:
    def __init__(self, scene: MicroScene, camera: Camera) -> None:
        self.scene = scene
        self.camera = camera
        self.device = scene.device
        self.window = spy.Window(
            width=camera.width,
            height=camera.height,
            title="neural.slang UTracer",
            resizable=False,
        )
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=camera.width, height=camera.height)
        self.window.on_keyboard_event = self._on_keyboard
        self.window.on_mouse_event = self._on_mouse
        self._blit_texture: Optional[spy.Texture] = None
        self._closed = False
        self._last_mouse = spy.float2(0, 0)
        self._drag_start = spy.float2(0, 0)
        self._camera_start = float3(0, 0, 0)
        self._dragging = False
        camera.reset_accumulator(scene.module)

    def run(self) -> None:
        try:
            while self.update():
                pass
        finally:
            self.close()

    def update(self) -> bool:
        if self._closed or self.window.should_close():
            return False
        self.window.process_events()
        if self.window.should_close():
            return False
        self._present(self.scene.render(self.camera))
        return True

    def _present(self, source: spy.Texture) -> None:
        image = self.surface.acquire_next_image()
        if image is None:
            return
        if (
            self._blit_texture is None
            or self._blit_texture.width != image.width
            or self._blit_texture.height != image.height
        ):
            self._blit_texture = self.device.create_texture(
                format=Format.rgba32_float,
                width=image.width,
                height=image.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            )
        self.scene.module.resample_rgb.dispatch(
            thread_count=uint3(image.width, image.height, 1),
            input=source,
            inputSampler=self.scene._sampler,
            output=self._blit_texture,
            outputSize=uint2(image.width, image.height),
        )
        command = self.device.create_command_encoder()
        command.blit(image, self._blit_texture)
        command.set_texture_state(image, spy.ResourceState.present)
        self.device.submit_command_buffer(command.finish())
        del image
        self.surface.present()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        window = self.window
        window.on_keyboard_event = None
        window.on_mouse_event = None
        self.device.wait()
        if self.surface.config:
            self.surface.unconfigure()
        self._blit_texture = None
        self.surface = None
        self.device.close()
        window.close()
        self.window = None

    def _on_keyboard(self, event: spy.KeyboardEvent) -> None:
        if event.type == spy.KeyboardEventType.key_press and event.key == spy.KeyCode.escape:
            self.window.close()

    def _on_mouse(self, event: spy.MouseEvent) -> None:
        if event.is_button_down() and event.button == spy.MouseButton.left:
            self._dragging = True
            self._drag_start = self._last_mouse
            self._camera_start = self.camera.pos
        elif event.is_button_up() and event.button == spy.MouseButton.left:
            self._dragging = False
        elif event.is_move():
            self._last_mouse = event.pos

        if self._dragging:
            delta = spy.float2(
                self._last_mouse.x - self._drag_start.x,
                self._last_mouse.y - self._drag_start.y,
            )
            original_direction = normalize(-self._camera_start)
            yaw = spy.math.atan2(original_direction.x, original_direction.z) - delta.x * 0.005
            pitch = spy.math.asin(original_direction.y) - delta.y * 0.005
            pitch = max(-1.5, min(1.5, pitch))
            direction = float3(
                spy.math.cos(pitch) * spy.math.sin(yaw),
                spy.math.sin(pitch),
                spy.math.cos(pitch) * spy.math.cos(yaw),
            )
            distance = spy.math.length(self._camera_start)
            self.camera.pos = -distance * direction
            self.camera.rot = quat_from_look_at(-normalize(self.camera.pos), float3(0, 1, 0))
            self.camera.recompute()
            self.camera.reset_accumulator(self.scene.module)

        if event.is_scroll():
            factor = 1.0 + event.scroll.y * 0.1
            distance = spy.math.length(self.camera.pos)
            distance = max(0.1, min(100.0, distance / factor))
            self.camera.pos = normalize(self.camera.pos) * distance
            self.camera.recompute()
            self.camera.reset_accumulator(self.scene.module)


def save_tonemapped(texture: spy.Texture, path: Union[str, PathLike[str]]) -> None:
    rgb = texture.to_numpy()[:, :, :3]
    rgb = rgb * 0.6
    rgb = np.clip(
        rgb * (2.51 * rgb + 0.03) / (rgb * (2.43 * rgb + 0.59) + 0.14),
        0.0,
        1.0,
    )
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(path)


def _compute_tangents(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    accumulated = np.zeros_like(positions)
    for triangle in faces:
        i0, i1, i2 = triangle
        dp1 = positions[i1] - positions[i0]
        dp2 = positions[i2] - positions[i0]
        duv1 = uvs[i1] - uvs[i0]
        duv2 = uvs[i2] - uvs[i0]
        denominator = duv1[0] * duv2[1] - duv2[0] * duv1[1]
        if abs(denominator) < 1e-8:
            continue
        tangent = (duv2[1] * dp1 - duv1[1] * dp2) / denominator
        accumulated[i0] += tangent
        accumulated[i1] += tangent
        accumulated[i2] += tangent

    tangents = np.zeros_like(positions)
    for index, (normal, tangent) in enumerate(zip(normals, accumulated)):
        tangent = tangent - normal * np.dot(normal, tangent)
        length = np.linalg.norm(tangent)
        if length <= 1e-8:
            axis = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
            tangent = np.cross(normal, axis)
            length = np.linalg.norm(tangent)
        tangents[index] = tangent / max(length, 1e-8)
    return tangents
