# Reference: `main.py` API

Complete technical description of every function in
`neuralnetwork/MNSIT-NEURAL-NETWORK/main.py`, the module that trains the MNIST
classifier and re-runs its forward pass in integer arithmetic.

The module has no package structure and exports nothing explicitly. Import it by
path, or run it directly with `python main.py`.

**Dependencies:** `numpy` (2.x tested), `keras` (3.x) plus a backend
(TensorFlow, PyTorch, or JAX).

---

## Module constants and layout

There are no module-level constants. Every shape and width is hardcoded inside
the functions that use it:

| Value | Where | Meaning |
|---|---|---|
| `60000` | `clean_data` | MNIST training set size |
| `10000` | `clean_data` | MNIST test set size |
| `784` | `clean_data`, `train_new_model` | Flattened 28x28 input |
| `10` | throughout | Hidden neurons, output classes |
| `0.5` | `clean_data` | Pixel binarization threshold |
| `2` | `to_fixed` default | Fractional bits in the fixed-point weights |
| `8` / `16` | `forward` | Accumulator widths passed to `check_overflow` |

---

## `clean_data(data)`

Converts raw MNIST arrays into the binary input format the integer forward pass
expects.

**Parameters**

| Name | Type | Constraints |
|---|---|---|
| `data` | `tuple` | Must be `((X_train, Y_train), (X_test, Y_test))`, the exact shape returned by `keras.datasets.mnist.load_data()`. `X_train` must contain exactly 60000 images and `X_test` exactly 10000, each 28x28 with values in `0..255`. |

**Returns** `(X_train, Y_train, X_test, Y_test)`

| Value | Shape | dtype | Domain |
|---|---|---|---|
| `X_train` | `(60000, 784)` | `int8` | `{0, 1}` |
| `Y_train` | `(60000, 10)` | `float64` | one-hot |
| `X_test` | `(10000, 784)` | `int8` | `{0, 1}` |
| `Y_test` | `(10000, 10)` | `float64` | one-hot |

**Transformation, in order**

1. Reshape each image from 28x28 to a flat 784-vector.
2. Cast to `float32` and divide by 255, giving values in `[0, 1]`.
3. Binarize: `1` where the normalized value is `>= 0.5`, else `0`. In raw
   terms, a pixel survives if its original byte value is `>= 128`. Cast to
   `int8`.
4. One-hot encode the labels to width 10 via `keras.utils.to_categorical`.

**Constraints**

The reshape sizes are literals, not derived from the input. Passing any dataset
that is not exactly 60000/10000 images raises
`ValueError: cannot reshape array of size N into shape (60000,784)`.

Binarization is lossy and irreversible. Grayscale detail is gone by the time any
other function sees the data, including `train_new_model`. The model is trained
on binary images, which is what makes the integer forward pass in
[`forward`](#forwardmodel_name-iterations5000) faithful rather than an
approximation. See
[Explanation: integer inference](explanation-integer-inference.md).

---

## `train_new_model(model_name)`

Trains a fresh classifier on binarized MNIST, saves it to disk, and prints test
metrics.

**Parameters**

| Name | Type | Constraints |
|---|---|---|
| `model_name` | `str` | Output path. The extension selects the format: `.keras` (native Keras 3) or `.h5` (legacy). Any other extension raises `ValueError: Invalid filepath extension for saving`. Relative paths resolve against the current working directory. |

**Returns** `None`. The trained model is written to `model_name` as a side
effect and then discarded.

**Architecture built**

```
Input(784)
  -> Dense(10)        kernel (784, 10)  bias (10,)   7850 params
  -> Activation(relu)
  -> Dense(10)        kernel (10, 10)   bias (10,)    110 params
  -> Activation(softmax)
                                            total    7960 params
```

Activations are separate `Activation` layers rather than `activation=` arguments
on the `Dense` layers. This is load-bearing: it fixes the layer indices that
[`forward`](#forwardmodel_name-iterations5000) uses (`layers[0]` and
`layers[2]` are the two `Dense` layers). Collapsing them into `Dense(10,
activation='relu')` changes the indices to `0` and `1` and breaks `forward`.

**Training configuration**

| Setting | Value |
|---|---|
| Loss | `categorical_crossentropy` |
| Optimizer | `adam` (Keras defaults, learning rate 0.001) |
| Metrics | `accuracy` |
| Batch size | `128` |
| Epochs | `15` |
| Validation split | `0.1` (54000 train / 6000 validation) |

**Output**

Keras prints a per-epoch progress bar to stdout during `fit` (422 steps per
epoch, from 54000 images at batch size 128). After saving, the function
evaluates on the test set with `verbose=2` and prints two lines:

```
Test Loss 0.27068477869033813
Test Accuracy 0.9229999780654907
```

Measured on a CPU-only TensorFlow 2.x backend: all 15 epochs plus evaluation
complete in about 6 seconds.

These are the *float* metrics for the Keras model. They are the ceiling for what
the integer forward pass can reach, not a prediction of it. The shipped model
scores 0.9237 in float and 85.44 percent through the integer pass; see
[How to tune the fixed-point format](howto-tune-fixed-point.md).

---

## `check_overflow(x, num_bits)`

Prints a warning if a value falls outside the signed two's-complement range of a
given bit width.

**Parameters**

| Name | Type | Constraints |
|---|---|---|
| `x` | numeric | Any Python or NumPy integer. |
| `num_bits` | `int` | Width of the hypothetical register. |

**Returns** `None`. Prints the literal string `overflow detected` to stdout when
`x > 2**(num_bits-1) - 1` or `x < -(2**(num_bits-1))`.

For `num_bits=8` the accepted range is `-128..127`; for `num_bits=16` it is
`-32768..32767`.

> **This function never fires as currently called.** Both call sites in
> `forward` pass a NumPy scalar that has *already* wrapped to the target width,
> so the value handed to `check_overflow` is by construction inside the range it
> tests. The wrap surfaces as a NumPy `RuntimeWarning: overflow encountered in
> scalar add` instead. Measured on the shipped model: a full 10000-image run at
> the default 2 fractional bits produces **zero** overflow events, so nothing is
> being missed today, but at 3 bits it produces 1370 and `check_overflow` still
> reports nothing. The mechanism and two fixes are in
> [How to tune the fixed-point format](howto-tune-fixed-point.md#making-overflow-detection-actually-work).

---

## `forward(model_name, iterations=5000)`

Loads a trained model, quantizes its weights to `int8`, and classifies test
images using only integer addition, comparison, and `max`. This is the function
that mirrors what a redstone build would do.

**Parameters**

| Name | Type | Default | Constraints |
|---|---|---|---|
| `model_name` | `str` | required | Path to a `.keras` file whose `layers[0]` and `layers[2]` are `Dense`. Resolved against the current working directory. |
| `iterations` | `int` | `5000` | Number of test images to classify. Values above 10000 are silently capped by the size of the test set. Values `<= 0` still classify one image, because the loop increments and checks `total` after the first pass. |

**Returns** `None`. Prints results to stdout.

### Weight extraction

| Variable | Source | Shape |
|---|---|---|
| `weights1` | `model.layers[0].get_weights()[0]` | `(784, 10)` |
| `biases1` | `model.layers[0].get_weights()[1]` | `(10,)` |
| `weights2` | `model.layers[2].get_weights()[0]` | `(10, 10)` |
| `biases2` | `model.layers[2].get_weights()[1]` | `(10,)` |

All four are passed through `to_fixed` and cast to `int8`.

### Nested: `to_fixed(float_value, bits_past_radix=2)`

Converts one float to a fixed-point integer. Multiplies by `2**bits_past_radix`,
rounds to the nearest integer, and returns it.

With the default of 2 fractional bits the scale is 4, so the stored integer is
`round(w * 4)` and one integer step represents 0.25. After the `int8` cast the
representable weight range is `-32.0` to `+31.75`.

The negative branch (`b = ~(abs(b)) + 1`) is a two's-complement negation written
out longhand. In Python it computes the value `b` already holds, so it is a
no-op. It documents the operation a redstone adder performs rather than changing
the result.

### Per-image algorithm

1. *Hidden layer.* For each of the 10 hidden neurons, walk the 784 input pixels
   and add `weights1.T[neuron][index]` to an accumulator wherever the pixel is
   `1`. Skip it where the pixel is `0`. Add `biases1[neuron]`. Because the input
   is binary, there is no multiply anywhere in this layer.
2. *ReLU.* `np.maximum(0, hidden_out)`, held as `int8`.
3. *Output layer.* For each of the 10 output neurons, accumulate
   `weights2.T[neuron][index] * value` over the 10 hidden activations, then add
   `biases2[neuron]`. This layer does multiply, but only 10 terms wide and with
   both operands small.
4. *Prediction.* `np.where(output_out == np.max(output_out), 1, 0)`, an
   argmax written as a one-hot vector. Softmax is skipped entirely: it is
   monotonic, so it cannot change which output is largest.
5. *Scoring.* Count the image as correct when the predicted one-hot vector is
   exactly equal to the label.

### Accumulator widths

The hidden accumulator is a NumPy `int8` and the output accumulator a NumPy
`int16`, both by type promotion rather than explicit declaration. They wrap on
overflow rather than growing. That wraparound is the point: it is what an 8-bit
redstone adder does. See
[Explanation: integer inference](explanation-integer-inference.md#accumulator-width-is-the-whole-game).

### Output

Progress lines every 100 images, each just the running count:

```
100
200
300
```

then one final line. Measured on the shipped `mnist_model.keras` over the full
test set:

```
Accuracy: 8544 / 10000 = 85.44%
```

The progress lines stop at `9900`. The loop breaks on `total >= iterations`
before reaching the print, so the final multiple of 100 is never shown: a
10000-image run emits 99 progress lines, not 100.

### Known edge cases

- *Ties count as wrong.* When two output neurons share the maximum value,
  `np.where` sets both to `1`. The result cannot equal a one-hot label, so the
  image is scored incorrect even if one of the tied neurons is the right answer.
  Integer outputs make ties far more likely than float outputs would. Measured:
  56 ties in a 10000-image run at the default 2 fractional bits, rising to 68
  per 2000 images when quantization is coarsened to 1 bit.
- *Overflow is silent.* See `check_overflow` above.
- *Path is relative.* `keras.saving.load_model` resolves `model_name` against
  the current working directory, so `forward` must be invoked from the directory
  holding the `.keras` file unless you pass an absolute path.
- *`iterations` below 1 still classifies one image.* `total` is incremented and
  compared after the first image is already processed, so `0` and negative
  values behave like `1`.

### Performance

The hidden layer runs 784 x 10 interpreted iterations per image, about 78
million for a full pass. Despite that, a complete 10000-image run measures **9
seconds wall clock** end to end on a laptop CPU, including interpreter startup,
the Keras import, and model load. The inner `if pixel == 1` guard skips about 87
percent of the additions: only 13.4 percent of pixels survive binarization
across the test set, since MNIST images are mostly dark.

---

## `main()`

Entry point, executed under `if __name__ == '__main__':`.

```python
def main():
    model_name = 'mnist_model.keras'
    # train_new_model(model_name)
    forward(model_name, 10000)
```

The training call is commented out by design: the repository ships a trained
`mnist_model.keras`, so the default action is to evaluate it, not to overwrite
it. To retrain, see
[How to retrain the model](howto-retrain-the-model.md).

---

## Repository artifacts

| Path | What it is |
|---|---|
| `neuralnetwork/MNSIT-NEURAL-NETWORK/main.py` | The module described above. All real code lives here. |
| `neuralnetwork/MNSIT-NEURAL-NETWORK/mnist_model.keras` | Trained weights, 112 KB. Native Keras 3 zip archive containing `config.json`, `metadata.json`, and `model.weights.h5`. Its saved architecture matches `train_new_model` exactly. |
| `main.py` (repository root) | A one-line `print("hello world")` left over from repository setup. Not part of the project. |

---

## Related

- [Tutorial: your first run](tutorial-first-run.md) walks through executing this API end to end.
- [How to retrain the model](howto-retrain-the-model.md)
- [How to tune the fixed-point format](howto-tune-fixed-point.md)
- [Explanation: why integer inference](explanation-integer-inference.md)
- [Explanation: mapping the network to redstone](explanation-redstone-mapping.md)
