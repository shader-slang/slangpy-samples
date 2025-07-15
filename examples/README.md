# SlangPy Examples

Welcome to SlangPy examples! Here you'll find well written and documented examples of how to use SlangPy, plus any sample code from the [documentation](https://slangpy.shader-slang.org).

## Standalone Samples

| Name                                                     | Output                                                            | Description                                                                                                                                  |
|----------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| [signed_distance_field](signed_distance_field/readme.md) | <img src="signed_distance_field/output.png" height="100">         | Use of SlangPy to generate a signed distance field and execute multiple eikonal passes to obtain accurate distance away from the zero point. |
| [ray-casting](ray-casting/README.md)                     | <img src="ray-casting/thumb.jpg" height="100">                    | Simple kernel that uses ray-casting to render a textured quad                                                                                |
| [toy-restir](toy-restir/README.md)                       | <img src="toy-restir/toy-restir.png" height="100">                | Uses ReSTIR to render the view of an animated 2D toy scene                                                                                   |
| [Simplified Splatting](simplified-splatting/README.md)   | <img src="simplified-splatting/simple-splat-ex.png" height="100"> | A simplified implementation of 3D Gaussian Splatting for educational purposes                                                                |


## Documentation Samples

| Name        | Description |
|-------------|-------------|
| [first_function](https://slangpy.shader-slang.org/en/latest/src/basics/firstfunctions.html) | Most basic use of SlangPy to call a single function. |
| [return_type](https://slangpy.shader-slang.org/en/latest/src/basics/firstfunctions.html) | Most basic use of different return types for a function. |
| [buffers](https://slangpy.shader-slang.org/en/latest/src/basics/buffers.html) | NDBuffer creation/use. |
| [textures](https://slangpy.shader-slang.org/en/latest/src/basics/textures.html) | Simple manipulation of texture data using SlangPy. |
| [nested](https://slangpy.shader-slang.org/en/latest/src/basics/nested.html) | Shows how to use Python dictionaries to pass nested data in SOA form. |
| [type_methods](https://slangpy.shader-slang.org/en/latest/src/basics/typemethods.html) | Use of InstanceList, InstanceBuffer and get_this to invoke methods of a type. |
| [broadcasting](https://slangpy.shader-slang.org/en/latest/src/basics/broadcasting.html) | Examples of how broadcasting works in SlangPy. |
| [mapping](https://slangpy.shader-slang.org/en/latest/src/basics/mapping.html) | Examples of how to use the `.map` modifier to map input dimensions to different call dimensions. |
| [autodiff](https://slangpy.shader-slang.org/en/latest/src/autodiff/autodiff.html) | Basic auto-diff examples, demonstrating how to evaluate a polynomial then run a backwards pass to calculate its gradients. |
| [pytorch](https://slangpy.shader-slang.org/en/latest/src/autodiff/pytorch.html) | Example of use of PyTorch integration to evaluate and back propagate through a polynomial. |
| [generators](https://slangpy.shader-slang.org/en/latest/src/generators/generators.html) | Demonstrates use of call id, thread id and grid generators. |
