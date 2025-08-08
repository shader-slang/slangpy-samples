# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np
from slangpy.types import call_id
import ast
import argparse
from typing import Optional, List, Dict

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

    def get_this(self) -> Dict:
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


def rm_lf(v : str) -> str:
    """ Helper function for logging; removes line feeds """
    return str(v.to_numpy()).replace('\n',',')

def main(
        log_values : bool,
        randomize_initial_params : bool,
        initial_params : list,
        reference_vertices : list,
        reference_dims : int,
        learning_rate : float,
        max_epochs : int,
        close_after : bool):
    """
    Main function that:
    1. Creates the App window, sets up the GPU device and loads the Slang shader module
    2. Rasterizes a triangle with predetermined vertices into a reference Tensor
    3. Initialize a Tensor of 3 vertex coordinates.
    4. In a loop, call slang to compute gradients for the coordinates, and then a second
       time to optimize the coordinates.

    Args:
        log_values(bool): Log primal and gradients each epoch?
        randomize_initial_params(bool): Randomize initial parameters?
        initial_params(list): List of 3 2D vertices.
        ref_vertices(str): String containing a list of 2D coordinates for the reference triangle
        learning_rate(float): Learning rate for gradient descent
        max_epochs(int): Maximum number of epochs
        close_after(bool): Close after reaching max_epochs
    """

    # Create app and load the rasterizer2d shader.
    app = App()
    rasterizer2d = spy.Module.load_from_file(app.device, "rasterizer2d.slang")

    # Setup the camera with the app's frame dimensions.
    app_camera = Camera(app)

    # Create a tensor to store the reference image.
    reference = spy.Tensor.zeros(app.device, dtype=spy.float4,
                                 shape=(reference_dims,reference_dims))

    # Create a second camera for the reference image.
    # This is necessary because the size of the framebuffer may be different.
    ref_camera = Camera(reference)

    # Call the rasterize function in Slang, passing the camera and vertices
    # array. We also use call_id() to pass the pixel coordinate within the target's
    # width and height, and set result to render the output color returned by
    # the rasterize function.
    #
    # Unlike the fwd-rasterize version, this function calculates per-pixel values
    # to determine if a pixel is inside the triangle, so the output values can be
    # used for training.
    rasterizer2d.rasterize(ref_camera, reference_vertices, call_id(), _result=reference)

    # Initialize an array of vertices to optimize.
    vertices_primal = spy.Tensor.zeros(app.device, dtype=spy.float2, shape=(3,))
    if randomize_initial_params:
        vertices_primal.copy_from_numpy(np.random.rand(3,).astype("float32") * 2 - 1)
    else:
        vertices_primal.copy_from_numpy(np.array(initial_params).astype("float32"))

    # Initialize an array of gradients for use in vertices optimization.
    # For convenience, we're using a Tensor here instead of an array, so
    # we can use the AtomicTensor type to make accumulating gradients
    # atomic in the slang shader.
    vertices_grad = spy.Tensor.zeros(app.device, dtype=spy.float2, shape=(3,))

    curr_iter = 0
    stay_open = True
    while app.process_events() and stay_open:
        if curr_iter < max_epochs:
            if log_values:
                print(f"primal: {rm_lf(vertices_primal)}")

            # Generate gradients using every pixel in the reference image
            rasterizer2d.generate_gradients(
                ref_camera, vertices_primal, vertices_grad, call_id(), reference)

            # Reminder: gradient totals have not yet been scaled down by # of samples
            if log_values:
                print(f"grad:   {rm_lf(vertices_grad)}")

            # Update vertices based on the generated gradients
            rasterizer2d.optimize(
                vertices_primal, vertices_grad, learning_rate, float(reference.element_count))
        else:
            if curr_iter == max_epochs:
                print(f"Completed {max_epochs} epochs")
                if close_after:
                    stay_open = False

        curr_iter = curr_iter + 1

        v = vertices_primal.to_numpy()
        vertices = [
            spy.float2(v[0][0], v[0][1]),
            spy.float2(v[1][0], v[1][1]),
            spy.float2(v[2][0], v[2][1]),
        ]

        rasterizer2d.rasterize(app_camera, vertices, call_id(), _result=app.output)

        app.present()

    print(f"Completed iterations: {min(curr_iter, max_epochs)}")
    print(f"Final parameters: {rm_lf(vertices_primal)}")


def eval_vertices_str(name : str, triangle : str, force_ccw : bool) -> Optional[List[spy.float2]]:
    """
    Evaluate a string of triangle vertices and validate its contents. The string should
    contain a 3x 2D vertices that can be cast to slangpy.float2. The vertices are reordered
    if necessary to produce a counter-clockwise winding order. Returns None on error.

    Args:
        name(str): Name of the parameter (for errors)
        triangle(str): String containing a list of 3 2D vertices
        force_cw(bool): Force counter-clockwise winding order?

    Returns:
        list | None: 3 slangpy.float2 vertices from the input triangle, or None on error.
    """
    # evaluate the string into a list of 2D vertices
    try:
        tri = ast.literal_eval(triangle)
        if len(tri) != 3:
            return None
        out = []
        for v in tri:
            out.append(spy.float2(v))
    except Exception as e:
        print(f"Error while parsing {name}")
        return None
    if not force_ccw:
        return out
    # ensure vertices are in counter-clockwise winding order
    AB = spy.float3(out[1],0) - spy.float3(out[0],0)
    AC = spy.float3(out[2],0) - spy.float3(out[0],0)
    cross = spy.math.cross(AB,AC)
    if cross.z < 0:
        print(f"Reordering {name} vertices to counter-clockwise winding order")
        return [ out[0], out[2], out[1] ]
    else:
        return out

def validate_gt(typ, value : int, lower_bound : int):
    v = typ(value)
    if v < lower_bound:
        raise argparse.ArgumentTypeError(f"Argument must be greater than {lower_bound}")
    return v

if __name__ == "__main__":

    desc = """
    'soft-rasterizer' is a SlangPy example. Demonstrates optimizing vertex parameters using
    gradient diffusion to match those of a previously rasterized reference soft-edged
    triangle. Many parameters for the vertices and gradient descent are exposed as command
    line arguments for experimentation, but it's easy to provide values that do not resolve.
    (Defaults should resolve.)
    """
    epilog = """
    Keyboard bindings: F1=Send output to tev, F2=Generate "screenshot.png", Esc=Quit
    """
    parser = argparse.ArgumentParser( description=desc, epilog=epilog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', "--log", action='store_true',
                        help="log the primal and gradient values to console")
    parser.add_argument('-r', "--random", action='store_true',
                        help="randomize initial vertex parameters; overrides initial vertex parameters")

    # Initial Parameters:  A triangle that covers the entire display.
    # Covers the region (-1,-1) to (1,1).
    #
    # a (-1, 1) *-------* c (3, 1)
    #           |   | /
    #           |---/ 1,-1
    #           | /
    # b (-1,-3) *
    parser.add_argument('-p', "--params", type=str,
                        default="(-1, 1), (-1, -3), (3, 1)",
                        help="initial vertex parameters, as a string")

    # Reference/Target: A right-angled triangle. (D=0.75)
    #
    # a (-D, D) *
    #           |\
    #           | \
    #           |  \
    # b (-D,-D) *---* c (D,-D)
    parser.add_argument('-t', "--target", type=str,
                        default="(-0.75, 0.75), (-0.75, -0.75), (0.75, -0.75)",
                        help="target/reference triangle vertices, as a string")
    parser.add_argument('-d', "--dims", type=lambda x: validate_gt(int, x, 64),
                        default=256,
                        help="dimensions of the reference raserized image, DxD")

    parser.add_argument("-g", "--learn", type=lambda x: validate_gt(float, x, 0.0000001),
                        default=0.01,
                        help="learning rate for gradient descent")
    parser.add_argument('-m', '--max', type=int, default=4000,
                        help="maximum iterations")
    parser.add_argument('-c', "--close", action='store_true',
                        help="close window after reaching maximum iterations")

    args = parser.parse_args()

    params = eval_vertices_str("initial vertex parameters", args.params, force_ccw=False)
    target = eval_vertices_str("target/reference triangle vertex parameters", args.target, force_ccw=True)
    if not params or not target:
        exit(-1)

    main(args.log, args.random, params, target, args.dims, args.learn, args.max, args.close)

