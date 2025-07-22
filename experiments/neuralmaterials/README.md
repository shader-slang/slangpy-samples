# Neural Materials

## About

This example shows how to implement and train a simple neural material in Slang and SlangPy to fit a reference material with a neural network.

This implementation is loosely based on the [Real-Time Neural Appearance Models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) work of Zeltner et al. The network first uses an encoder to transform the parameters of the reference material into a latent code, and then a decoder to evaluate the material response for the latent code and a given set of directions. For simplicity, we're omitting a few components of the original paper, such as the shading frame encoding, input encodings and LOD.

After training is finished, the encoder can be dropped by "baking" the reference material textures into a latent texture. This is not currently implemented.

This example uses the neural networks library from the SlangPy [Neural Texture Sample](https://github.com/shader-slang/slangpy-samples/tree/main/experiments/neuralnetwork), and you can read through its documentation to learn more about neural networks in SlangPy.

## How to Run

First install slangpy with `pip install slangpy` (see also the [SlangPy docs](https://slangpy.shader-slang.org/en/latest/installation.html)). Then run the example with `python main.py`.

You should see a rendering of the neural material evolving live as training progresses. We recommend running on a GPU with cooperative vector support for fast training.

After a few thousand iterations, you should see an output like this:

![](./example-image.jpg)
