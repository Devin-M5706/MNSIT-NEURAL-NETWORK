# Tutorial: your first run

In the next ten minutes you will classify 10000 handwritten digits using nothing
but 8-bit integer addition, watch a single digit travel through the network one
neuron at a time, and see exactly where the accuracy goes when you take floating
point away.

By the end you will understand what the redstone build has to do, because you
will have watched the numbers it has to produce.

## What you'll need

- **Python 3.10 or newer.** Check with `python --version`.
- **About 200 MB of disk** for Keras and a backend.
- **Internet access** the first time, to download MNIST (11 MB, cached
  afterward).
- **No GPU, no prior ML experience.** The whole model is 7960 parameters and the
  full evaluation takes 9 seconds.

## Step 1: Install the dependencies

```bash
pip install numpy keras tensorflow-cpu
```

`tensorflow-cpu` is the backend Keras runs on. PyTorch or JAX work equally well
if you already have one.

## Step 2: Run the simulator

The code and the trained weights live in a subdirectory. You have to run from
there, because the model file is loaded by a relative path.

```bash
cd neuralnetwork/MNSIT-NEURAL-NETWORK
python main.py
```

You will see a count climbing in hundreds, then the result:

```
100
200
300
...
9800
9900
Accuracy: 8544 / 10000 = 85.44%
```

**That is the whole thing working.** 85.44 percent of ten thousand handwritten
digits, classified without a single floating point operation at inference time.
Every number involved fit in 8 or 16 bits, which is the point: those are widths
you can build out of redstone.

(The count stops at 9900 rather than 10000. The loop breaks before its last
progress print. All 10000 images were classified.)

## Step 3: Look at what the network actually sees

The input is not a grayscale image. `clean_data` throws away every shade and
keeps one bit per pixel: lit or unlit. Print the first test image:

```bash
python -c "
import keras, numpy as np, main
_, _, X_test, Y_test = main.clean_data(keras.datasets.mnist.load_data())
for row in X_test[0].reshape(28, 28):
    print(''.join('##' if p else '  ' for p in row))
print('label:', int(np.argmax(Y_test[0])), '| lit pixels:', int(X_test[0].sum()), 'of 784')
"
```

```
              ######
            ##############################
                    ########################
                                      ####
                                      ####
                                    ####
                                  ######
                                  ####
                                ######
                                ####
                                ####
                              ####
                            ######
                          ######
                          ####
                        ####
                      ######
                      ######
                      ######
                      ####

label: 7 | lit pixels: 71 of 784
```

A seven, in 71 lit pixels out of 784. That grid is buildable: 784 lamps, each on
or off, and you could set them by hand.

This is also the single most important optimization in the project. Because
every input is 0 or 1, multiplying a weight by a pixel is not multiplication. It
is a decision to add the weight or skip it. The 784-input layer needs adders and
gates, and **no multipliers at all**.

## Step 4: Follow one digit through the network

Now watch that seven get classified, stage by stage. This snippet reproduces
what `forward()` does internally for a single image:

```bash
python -c "
import keras, numpy as np, main
_, _, X_test, Y_test = main.clean_data(keras.datasets.mnist.load_data())
m = keras.saving.load_model('mnist_model.keras')

to_fixed = lambda v, b=2: int(round(v * (2 ** b)))
w1 = np.vectorize(to_fixed)(m.layers[0].get_weights()[0]).astype('int8')
b1 = np.vectorize(to_fixed)(m.layers[0].get_weights()[1]).astype('int8')
w2 = np.vectorize(to_fixed)(m.layers[2].get_weights()[0]).astype('int8')
b2 = np.vectorize(to_fixed)(m.layers[2].get_weights()[1]).astype('int8')

X = X_test[0]
hidden = []
for n in range(10):
    acc = 0
    for i, pixel in enumerate(X):
        if pixel == 1:
            acc += w1.T[n][i]
    hidden.append(int(acc + b1.T[n]))
h = np.maximum(0, np.array(hidden).astype('int8')).astype('int8')

out = []
for n in range(10):
    acc = 0
    for i, value in enumerate(h):
        acc += w2.T[n][i] * np.int16(value)
    out.append(int(acc + b2.T[n]))

print('quantized weight range :', int(w1.min()), 'to', int(w1.max()))
print('hidden, before ReLU    :', hidden)
print('hidden, after ReLU     :', [int(v) for v in h])
print('output scores          :', out)
print('argmax:', int(np.argmax(out)), '| true label:', int(np.argmax(Y_test[0])))
"
```

```
quantized weight range : -4 to 3
hidden, before ReLU    : [13, 24, 9, 41, 7, 12, 47, -6, 31, 5]
hidden, after ReLU     : [13, 24, 9, 41, 7, 12, 47, 0, 31, 5]
output scores          : [-39, -179, -1, 119, -222, 5, -295, 129, -39, -73]
argmax: 7 | true label: 7
```

Read that output carefully, because every line is a circuit:

- **`quantized weight range: -4 to 3`.** Every weight in the 784-wide layer is a
  small signed integer. A weight of `3` means 0.75, because two bits sit past
  the binary point and the scale is 4. Fixed point turns fractional weights into
  ordinary integer addition.
- **`hidden, before ReLU`.** Ten sums, each built by adding up the weights of the
  71 lit pixels. No multiplication happened.
- **`after ReLU`.** Neuron 7 went from `-6` to `0`. That is the entire
  activation function: test the sign bit, and if it is set, output zero. One
  inverter and a gate.
- **`output scores`.** The winner is index 7 with `129`. There is no softmax
  anywhere. Softmax cannot change which number is biggest, so classification
  never needs it, and skipping it removes ten exponentials and a division from
  the circuit.

## Step 5: Break it, and see why the widths matter

The accumulator in step 4 is 8 bits wide. Weights are quantized with 2
fractional bits. Both of those are choices, and they are the *same* choice.

Open `main.py`, find `to_fixed` inside `forward()`, and change the default:

```python
def to_fixed(float_value, bits_past_radix=4):
```

Run it again:

```bash
python main.py
```

```
Accuracy: 2865 / 10000 = 28.65%
```

Accuracy fell off a cliff, from 85.44 percent to 28.65. Finer weights should be
*better*. What happened is that quadrupling the scale quadrupled every partial
sum, and the 8-bit accumulator started wrapping around: 46538 times over the
test set. When an accumulator wraps, a large positive sum becomes negative, ReLU
clamps it to zero, and the neuron goes dark.

Now try 3 instead of 4:

```
Accuracy: 8667 / 10000 = 86.67%
```

Better than the shipped default. Three fractional bits buys enough weight
resolution to be worth the 1370 overflows it causes.

**Set it back to `2` before moving on.** That is the setting where the
accumulator provably never overflows, which is worth more in a physical circuit
than 1.2 points of accuracy.

## What you built

You ran a neural network that a Minecraft redstone machine could execute
literally. Concretely, you now know:

- The input is **784 bits**, so the biggest layer needs zero multipliers.
- The hidden layer is **10 accumulators, 8 bits wide**, fed by gated adders.
- ReLU is **one sign-bit test**.
- The output layer is where the only 100 multipliers live, accumulating in **16
  bits**.
- The prediction is **an argmax comparator tree**, with no softmax.
- The cost of all of this is **92.37% in float against 85.44% in integers**, and
  the binding constraint is accumulator width, not weight precision.

### Where to go next

- **Understand the reasoning:**
  [Why this network runs on integers](explanation-integer-inference.md) explains
  each choice above and what it traded away.
- **See the circuits:**
  [Mapping the network to redstone](explanation-redstone-mapping.md) turns each
  stage into a structure, with the open questions named honestly.
- **Tune it properly:**
  [How to tune the fixed-point format](howto-tune-fixed-point.md) has the full
  measured sweep and shows how to count overflow instead of guessing at it.
- **Train your own weights:**
  [How to retrain the model](howto-retrain-the-model.md), about 6 seconds of CPU.
- **Look up any function:** [Reference: `main.py` API](reference-api.md).
