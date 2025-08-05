# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np
from slangpy.types import call_id
import time
import argparse

class Camera:
    """
    This Camera class contains data for the Camera struct from rasterizer2d.slang, setting
    the origin, scale, and frame size that the Slang camera struct expects. The frame size is
    retrieved when needed from either an app window or Tensor to avoid redundant state.

    Attributes:
        origin (slangpy.float2): origin in worldspace
        scale (slangpy.float2): scale in worldspace
        frame_source (App|spy.Tensor): associated App or Tensor
    """

    def __init__(self, frame_source: App | spy.Tensor):
        """
        Initialize the Camera with origin at (0,0) and scale (1,1), and use frame
        dimensions from either an App window or slangpy.Tensor.

        Args:
            frame_source (App|spy.Tensor): Frame to retrieve dimensions from
        """
        if not (isinstance(frame_source, App) or isinstance(frame_source, spy.Tensor)):
            raise Exception("Camera only supports App or slangpy.Tensor as frame_source.")
        if isinstance(frame_source, spy.Tensor) and len(frame_source.shape) != 2:
            raise Exception("slangpy.Tensor must be 2D")
        self.origin = spy.float2(0.0, 0.0)
        self.scale = spy.float2(1.0, 1.0)
        self.frame_source = frame_source

    def get_this(self):
        """
        Return a dict with the Camera attributes mapped to names that the
        Slang struct expects.
        """
        if isinstance(self.frame_source, App):
            dim = spy.float2(self.frame_source._window.width, self.frame_source._window.height)
        elif isinstance(self.frame_source, spy.Tensor):
            dim = spy.float2(self.frame_source.shape[0], self.frame_source.shape[1])
        return {
            "origin": self.origin,
            "scale": self.scale,
            "frameDim": dim,
            "_type": "Camera",
        }


def rm_lf(v):
    """ Helper function for logging; removes line feeds """
    return str(v.to_numpy()).replace('\n',',')

def main(init_predefined_triangle : bool = False, log_values : bool = False):
    """
    Main function that:
    1. Creates the App window, sets up the GPU device and loads the Slang shader module
    2. Rasterizes a triangle with predetermined vertices into a reference Tensor
    3. Initialize a Tensor of 3 vertex coordinates.
    4. In a loop, call slang to compute gradients for the coordinates, and then a second
       time to optimize the coordinates.
    """

    # Create app and load the rasterizer2d shader.
    app = App()
    rasterizer2d = spy.Module.load_from_file(app.device, "rasterizer2d.slang")

    # Buffer of 3 vertex positions for the triangle. (D=0.75)
    # a (-D, D) *
    #           |\
    #           | \
    #           |  \
    # b (-D,-D) *---* c (D,-D)
    ref_vertices = [
        spy.float2(-0.75, 0.75),
        spy.float2(-0.75, -0.75),
        spy.float2(0.75, -0.75),
    ]

    # Setup the camera with the app's frame dimensions.
    app_camera = Camera(app)

    # Create a tensor to store the reference image.
    reference = spy.Tensor.zeros(app.device, dtype=spy.float4, shape=(64,64))

    # Create a second camera for the reference image.
    ref_camera = Camera(reference)

    # Call the rasterize function in Slang, passing the camera and vertices
    # array. We also use call_id() to pass the pixel coordinate within the target's
    # width and height, and set result to render the output color returned by
    # the rasterize function.
    #
    # Unlike the fwd-rasterize version, this function calculates per-pixel values
    # to determine if a pixel is inside the triangle, so the output values can be
    # used for training.
    rasterizer2d.rasterize(ref_camera, ref_vertices, call_id(), _result=reference)

    # Initialize an array of vertices to optimize. There are two ways we can
    # initialize this; use a predefined triangle, in this case one that covers
    # the entire display, or with random vertices.
    #
    # A triangle that covers the entire display would need to cover the region
    # (-1,-1) to (1,1), so let's use the following triangle:
    #
    # a (-1, 1) *-------* c (3, 1)
    #           |   | /
    #           |---/ 1,-1
    #           | /
    # b (-1,-3) *
    if init_predefined_triangle:
        covering_triangle = [
            [-1, 1],
            [-1,-3],
            [ 3, 1],
        ]
        vertices_primal = spy.Tensor.zeros(app.device, dtype=spy.float2, shape=(3,))
        vertices_primal.copy_from_numpy(np.array(covering_triangle).astype("float32"))
    else:
        vertices_primal = spy.Tensor.zeros(app.device, dtype=spy.float2, shape=(3,))
        vertices_primal.copy_from_numpy(np.random.rand(3,).astype("float32") * 2 - 1)

    # Initialize an array of gradients for use in vertices optimization.
    # For convenience, we're using a Tensor here instead of an array, so
    # we can use the AtomicTensor type to make accumulating gradients
    # atomic in the slang shader.
    vertices_grad = spy.Tensor.zeros(app.device, dtype=spy.float2, shape=(3,))

    currIter = 0
    maxIter = 1000
    learning_rate = 0.01;
    while app.process_events():
        if currIter < maxIter:
            if log_values:
                print(f"+primal: {rm_lf(vertices_primal)}")
                print(f"0grad:   {rm_lf(vertices_grad)}")

            # Generate gradients using every pixel in the reference image
            rasterizer2d.generate_gradients(
                ref_camera, vertices_primal, vertices_grad, call_id(), reference)

            if log_values:
                print(f" primal: {rm_lf(vertices_primal)}")
                print(f"+grad:   {rm_lf(vertices_grad)}")

            # Update vertices based on the generated gradients
            rasterizer2d.optimize(
                vertices_primal, vertices_grad, learning_rate, float(reference.element_count))

            currIter = currIter + 1
        else:
            # Once we've hit the max iterations, keep updating the app window to show the result.
            time.sleep(0.25)

        v = vertices_primal.to_numpy()
        vertices = [
            spy.float2(v[0][0], v[0][1]),
            spy.float2(v[1][0], v[1][1]),
            spy.float2(v[2][0], v[2][1]),
        ]

        rasterizer2d.rasterize(app_camera, vertices, call_id(), _result=app.output)

        app.present()

    print(f"Completed iterations: {currIter}")
    print(f"Final vertices: {rm_lf(vertices_primal)}")


if __name__ == "__main__":
    """ main entry point """

    desc = """
    'soft-rasterizer' is a SlangPy example. Demonstrates optimizing vertices using gradient
    diffusion to match those of a previously rasterized soft-edged triangle.
    """
    parser = argparse.ArgumentParser( description=desc)
    parser.add_argument('-l', "--log", action='store_true',
                        help="log the primal and gradient values")
    parser.add_argument('-p', "--predef", action='store_true',
                        help="init to predefined triangle (otherwise, init to randomized values)")
    args = parser.parse_args()
    main(args.predef, args.log)

