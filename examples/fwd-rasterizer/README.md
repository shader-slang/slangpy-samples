# Forward Triangle Rasterizer

This example rasterizes a single triangle using per-sample evaluation of
whether pixel coordinates lie inside/outside a triangle's boundaries.

This the first half of a two-part example, showing the application of autodiff
to recover vertex parameters for a simple rasterization example. Refer to the
soft triangle rasterizer example (soft-rasterizer) for the second part.

## Overview

This example:

* Creates the App window, sets up the GPU device and loads the Slang shader module
* Prepares a buffer with vertices for a single tringle
* Rasterizes the triangle using the Slang shader module

## Output Example

![Output visualization](output.png)

_Visualization showing the output triangle._

## Prerequisites

- SlangPy
- [tev](https://github.com/Tom94/tev) (optional)

## Keyboard Bindings

- ``F1`` - Send the output texture in its native format to a running ``tev`` process.
- ``F2`` - Write a screenshot from output texture to "screenshot.png" (in 8-bit format).
- ``Esc`` - Quit

## Usage

Call the main.py script to display the triangle. This example is intentionally
kept simple and takes no parameters.

