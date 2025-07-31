

This example takes the forward rasterizer, and turns it into a soft rasterizer
with fuzzy edges instead of boolean inside/outside.


In broad strokes, what we want to do:

- Generate a reference image
- Assign initial parameters (random triangle vertices)
- Loop epochs
    - For each pixel:
        - Get reference sample
        - Calculate sample based on parameters
        - Calculate Loss
        - bwd_diff
        - Accumulate gradients
    - Update parameters
- Display final triangle

We're accumulating gradients from multiple samples into three vertices, so this
will be happening across multiple threads. That means we'll need to use an
AtomicTensor to collect the results.



