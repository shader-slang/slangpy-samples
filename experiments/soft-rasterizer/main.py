# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from app import App
import slangpy as spy
import numpy as np
from slangpy.types import call_id

# This Camera class mimics the Camera struct from rasterizer2d.slang, setting
# the origin, scale, and window size that the Slang camera struct expects. The
# window size is retrieved from the app window to avoid redundant state.


class Camera:
    # Origin is at 0,0 with 1,1 scale.
    def __init__(self, app):
        self.o = spy.float2(0.0, 0.0)
        self.scale = spy.float2(1.0, 1.0)
        self.app = app

    # Return a dict with the class variables mapped to the names that the
    # Slang struct expects.
    def get_this(self):
        return {
            "o": self.o,
            "scale": self.scale,
            "frameDim": spy.float2(self.app._window.width, self.app._window.height),
            "_type": "Camera"
        }


# Create app and load the rasterizer2d shader.
app = App()
rasterizer2d = spy.Module.load_from_file(app.device, "rasterizer2d.slang")

# Buffer of 3 vertex positions for the triangle.
vertices = [
    spy.float2(-0.75,  0.75),
    spy.float2(-0.75, -0.75),
    spy.float2(0.75, -0.75)
]

# Setup the camera with the app's frame dimensions.
camera = Camera(app)

# Run the app.
currIter = 0
maxIter = 400
while app.process_events():
    # Call the rasterize function in Slang, passing the camera and vertices
    # array. We also use call_id to pass the pixel coordinate within window
    # width and height, and set result to render the output color returned by
    # the rasterize function.
    #
    # Unlike the fwd-rasterize version, this function calculates a per-pixel
    # probability values to determine if a pixel is inside the triangle, and
    # can be more easily extended to a trained version.
    rasterizer2d.rasterize(camera, vertices, call_id(), _result=app.output)
    app.present()
