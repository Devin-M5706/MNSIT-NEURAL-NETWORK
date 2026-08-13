# MNIST Neural Network Simulator (for Minecraft)

A neural network trained on the **MNIST handwritten digits dataset**, then
re-implemented so its entire forward pass runs on 8-bit and 16-bit integer
arithmetic. No floating point, no exponentials, no division. Everything it does
is something redstone can do.

The goal is to rebuild it inside **Minecraft**, in redstone.

Inspired by the work of [MattBatWings](https://www.youtube.com/@MattBatWings).
Project write-up on
[LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7314837008021950464/).

## Results

| | |
|---|---|
| Float model (Keras, float32) | **92.37%** on 10000 test images |
| Integer forward pass (int8 / int16) | **85.44%** on 10000 test images |
| Model size | 7960 parameters |
| Full evaluation runtime | about 9 seconds, CPU only |

Quantization costs just under 7 points. That gap is the price of a machine you
can actually build.

## Quickstart

```bash
pip install numpy keras tensorflow-cpu
cd neuralnetwork/MNSIT-NEURAL-NETWORK
python main.py
```

```
100
200
...
9900
Accuracy: 8544 / 10000 = 85.44%
```

Full walkthrough: [Tutorial: your first run](docs/tutorial-first-run.md).

## Architecture

```
Input      784 nodes    28x28, binarized to 1 bit per pixel
   |
Dense       10 nodes    int8 weights, 2 bits past the radix
   |
ReLU                    sign-bit test
   |
Dense       10 nodes    int16 accumulator
   |
Softmax                 training only, skipped at inference
```

| | |
|---|---|
| Input | 784 binary values, threshold 0.5 |
| Hidden layer | 10 neurons, ReLU |
| Output layer | 10 neurons, digits 0 through 9 |
| Loss (training) | Categorical crossentropy |
| Optimizer (training) | Adam, 15 epochs, batch size 128 |
| Inference arithmetic | Fixed-point int8 weights, int8 hidden accumulator, int16 output accumulator |
| Prediction | Argmax over raw output values |

Three choices make the whole thing buildable:

1. **Binary input.** Pixels are 0 or 1, so a weight is either added or skipped.
   The 784-wide layer needs **no multipliers**.
2. **ReLU, not sigmoid.** `max(0, x)` is one sign-bit test. Sigmoid would need a
   lookup table for `e^-x`.
3. **No softmax at inference.** Softmax is monotonic, so it cannot change which
   output is largest. Skipping it is exact, not an approximation, and removes
   ten exponentials and a division.

Reasoning in full:
[Why this network runs on integers](docs/explanation-integer-inference.md).

## Documentation

| Document | Type | What it covers |
|---|---|---|
| [Your first run](docs/tutorial-first-run.md) | Tutorial | Install to working result, then trace one digit through the network neuron by neuron |
| [How to retrain the model](docs/howto-retrain-the-model.md) | How-to | Train fresh weights and verify them through the integer pass |
| [How to tune the fixed-point format](docs/howto-tune-fixed-point.md) | How-to | Change the fractional bit count, detect overflow, measure the cost |
| [`main.py` API](docs/reference-api.md) | Reference | Every function, parameter, shape, dtype, and edge case |
| [Why integer inference](docs/explanation-integer-inference.md) | Explanation | The design reasoning, the trade-offs, and the measured 8-bit ceiling |
| [Mapping to redstone](docs/explanation-redstone-mapping.md) | Explanation | Each layer as a circuit, plus the open build questions |

## Repository layout

```
main.py                                       placeholder, not part of the project
neuralnetwork/MNSIT-NEURAL-NETWORK/
    main.py                                   all project code
    mnist_model.keras                         trained weights, 112 KB
docs/                                         documentation
```

## Technologies

- Python 3.10+, NumPy 2.x
- Keras 3.x with a TensorFlow, PyTorch, or JAX backend (training only)
- MNIST dataset, 28x28 grayscale, binarized
- Minecraft Java Edition, creative mode, for the redstone build

## Roadmap

The simulator is done. The build is not.

- [x] Train a classifier small enough to build
- [x] Re-run the forward pass in pure integer arithmetic
- [x] Verify accumulator widths against real overflow counts
- [ ] Pick serial versus parallel accumulation for the hidden layer
- [ ] Design the 8x8 multiplier used 100 times in the output layer
- [ ] Weight ROM layout
- [ ] Priority encoder for the argmax display, so ties light one lamp
- [ ] Build it

Current status and the honest gaps:
[Mapping the network to redstone](docs/explanation-redstone-mapping.md#what-the-simulator-does-not-answer).
