

This example takes the forward rasterizer, and turns it into a soft rasterizer
with fuzzy edges instead of boolean inside/outside.


In broad strokes, this example will:

- Generate a reference image
- Assign initial vertex parameters values (to be trained to the reference)
- Loop epochs:
    - For each pixel coordinate:
        - Get reference sample
        - Calculate a sample based on this epoch's vertex parameters
        - Calculate Loss versus the reference pixel
        - bwd_diff(loss)
        - Accumulate gradients
    - Optimize the vertex parameters using the gradients (gradient descent)
    - Display triangle

We're accumulating gradients from multiple samples into three vertices, so this
will be happening across multiple threads. That means we'll need to avoid
overwriting and invalidating the gradient values; for this, we'll use atomic
writes by using an AtomicTensor.



