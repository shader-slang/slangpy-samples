# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional
import slangpy as spy


class App:
    def __init__(
        self,
        device: spy.Device,
        title: str = "Neural Networks Example",
        width: int = 1024,
        height: int = 1024,
    ):
        super().__init__()

        # Create a window
        self._window = spy.Window(width=width, height=height, title=title, resizable=False)

        # Create a device with local include path for shaders
        self._device = device

        # Setup swapchain
        self.surface = self._device.create_surface(self._window)
        self.surface.configure(width=self._window.width, height=self._window.height)

        # Will contain output texture
        self._output_texture: spy.Texture = self.device.create_texture(
            format=spy.Format.rgba16_float,
            width=width,
            height=height,
            mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="output_texture",
        )

        # Store mouse pos
        self._mouse_pos = spy.float2()

        # Internal events
        self._window.on_keyboard_event = self._on_window_keyboard_event
        self._window.on_mouse_event = self._on_window_mouse_event
        self._window.on_resize = self._on_window_resize

        # Hookable events
        self.on_keyboard_event: Optional[Callable[[spy.KeyboardEvent], None]] = None
        self.on_mouse_event: Optional[Callable[[spy.MouseEvent], None]] = None

    @property
    def device(self) -> spy.Device:
        return self._device

    @property
    def window(self) -> spy.Window:
        return self._window

    @property
    def mouse_pos(self) -> spy.float2:
        return self._mouse_pos

    @property
    def output(self) -> spy.Texture:
        return self._output_texture

    def process_events(self):
        if self._window.should_close():
            return False
        self._window.process_events()
        return True

    def present(self):
        if not self.surface.config:
            return
        image = self.surface.acquire_next_image()
        if not image:
            return

        if (
            self._output_texture == None
            or self._output_texture.width != image.width
            or self._output_texture.height != image.height
        ):
            self._output_texture = self.device.create_texture(
                format=spy.Format.rgba16_float,
                width=image.width,
                height=image.height,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="output_texture",
            )

        command_encoder = self._device.create_command_encoder()
        command_encoder.blit(image, self._output_texture)
        command_encoder.set_texture_state(image, spy.ResourceState.present)
        self._device.submit_command_buffer(command_encoder.finish())

        del image
        self.surface.present()

    def _on_window_keyboard_event(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self._window.close()
                return
            elif event.key == spy.KeyCode.f1:
                if self._output_texture:
                    spy.tev.show_async(self._output_texture)
                return
            elif event.key == spy.KeyCode.f2:
                if self._output_texture:
                    bitmap = self._output_texture.to_bitmap()
                    bitmap.convert(
                        spy.Bitmap.PixelFormat.rgb,
                        spy.Bitmap.ComponentType.uint8,
                        srgb_gamma=True,
                    ).write_async("screenshot.png")
                return
        if self.on_keyboard_event:
            self.on_keyboard_event(event)

    def _on_window_mouse_event(self, event: spy.MouseEvent):
        if event.type == spy.MouseEventType.move:
            self._mouse_pos = event.pos
        if self.on_mouse_event:
            self.on_mouse_event(event)

    def _on_window_resize(self, width: int, height: int):
        self._device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height)
        else:
            self.surface.unconfigure()
