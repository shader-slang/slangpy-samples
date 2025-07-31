# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np
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
    app = App()
    rasterizer2d = spy.Module.load_from_file(app.device, "rasterizer2d.slang")

    # Buffer of 3 vertex positions for the triangle.
    ref_vertices = [
        spy.float2(-0.75, 0.75),
        spy.float2(-0.75, -0.75),
        spy.float2(0.75, -0.75),
    ]

    # Setup the camera with the app's frame dimensions.
    camera = Camera(app)

    # Create a tensor to store the reference image.
    #reference = spy.Tensor(dtype=float16)
    reference = spy.Tensor.zeros(app.device, dtype=spy.float4, shape=(64,64))

    # Call the rasterize function in Slang, passing the camera and vertices
    # array. We also use call_id to pass the pixel coordinate within window
    # width and height, and set result to render the output color returned by
    # the rasterize function.
    #
    # Unlike the fwd-rasterize version, this function calculates a per-pixel
    # probability values to determine if a pixel is inside the triangle, and
    # can be more easily extended to a trained version.
    rasterizer2d.rasterize(camera, ref_vertices, call_id(), _result=reference)

    # Initialize an array of vertices to optimize.
    vertices_primal = spy.Tensor.zeros(app.device, dtype=spy.float2, shape=(3,))
    vertices_primal.copy_from_numpy(np.random.uniform(0, 1, (6,)).astype("float32"))

    # Initialize an array of gradients for use in vertices optimization.
    # For convenience, we're using a Tensor so we can use the AtomicType
    # on the slang side.
    vertices_grad = spy.Tensor.zeros(app.device, dtype=spy.float2, shape=(3,))

    currIter = 0
    maxIter = 400
    learning_rate = 0.01;
    while app.process_events():
        if currIter < maxIter:
            # Generate gradients using every pixel in the reference image
            print(f"primal: {vertices_primal.to_numpy()}")
            print(f"grad:   {vertices_grad.to_numpy()}")
            rasterizer2d.generate_gradients(camera, vertices_primal, vertices_grad, call_id(), reference)

            # Update vertices based on the generated gradients
            print(f"primal: {vertices_primal.to_numpy()}")
            print(f"grad:   {vertices_grad.to_numpy()}")
            rasterizer2d.optimize(vertices_primal, vertices_grad, learning_rate)

            currIter = currIter + 1

        v = vertices_primal.to_numpy()
        vertices = [
            spy.float2(v[0][0], v[0][1]),
            spy.float2(v[1][0], v[1][1]),
            spy.float2(v[2][0], v[2][1]),
        ]
        print(vertices)
        rasterizer2d.rasterize(camera, vertices, call_id(), _result=app.output)
        app.present()


    if False:
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


if __name__ == "__main__":
    main()
