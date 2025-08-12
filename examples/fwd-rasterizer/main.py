# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from common.app import App
import slangpy as spy
from slangpy.types import call_id


class Camera:
    """
    This Camera class mimics the Camera struct from rasterizer2d.slang, setting
    the origin, scale, and window size that the Slang camera struct expects. The
    window size is retrieved from the app window to avoid redundant state.

    Attributes:
        origin (slangpy.float2): origin in worldspace
        scale (slangpy.float2): scale in worldspace
        app (App): assiciated App
    """

    def __init__(self, app: App):
        """
        Initialize the Camera with origin at (0,0) and scale (1,1)

        Args:
            app (App): App to retrieve window size from
        """
        self.origin = spy.float2(0.0, 0.0)
        self.scale = spy.float2(1.0, 1.0)
        self.app = app

    def get_this(self):
        """
        Return a dict with the Camera attributes mapped to names that the
        Slang struct expects.
        """
        return {
            "origin": self.origin,
            "scale": self.scale,
            "frameDim": spy.float2(self.app._window.width, self.app._window.height),
            "_type": "Camera",
        }


def main():
    """
    Main function that
    1. Creates the App window, sets up the GPU device and loads the Slang shader module
    2. Prepares a buffer with vertices for a single tringle
    3. Rasterizes the triangle using the Slang shader module
    """

    # Create app and load the rasterizer2d shader.
    app = App(
        title="fwd-rasterizer",
        width=1024,
        height=1024,
        device_type=spy.DeviceType.automatic,
        include_paths=[Path(__file__).parent],
    )
    rasterizer2d = spy.Module.load_from_file(app.device, "rasterizer2d.slang")

    # Buffer of 3 vertex positions for the triangle.
    vertices = [
        spy.float2(-0.75, 0.75),
        spy.float2(-0.75, -0.75),
        spy.float2(0.75, -0.75),
    ]

    # Setup the camera with the app's frame dimensions.
    camera = Camera(app)

    # Run the app.
    while app.process_events():
        # Call the rasterize function in Slang, passing the camera and vertices
        # array. We also use call_id to pass the pixel coordinate within window
        # width and height, and set result to render the output color returned by
        # the rasterize function.
        rasterizer2d.rasterize(camera, vertices, call_id(), _result=app.output)
        app.present()


if __name__ == "__main__":
    main()
