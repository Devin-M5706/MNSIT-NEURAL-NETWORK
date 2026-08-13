# How to retrain the model

Train a fresh set of weights on binarized MNIST and save them over
`mnist_model.keras`, then confirm the integer forward pass still works against
the new weights.

Do this when you change the architecture, the binarization threshold, or the
number of hidden neurons. You do **not** need to retrain to change the
fixed-point format; that happens at load time, so see
[How to tune the fixed-point format](howto-tune-fixed-point.md) instead.

## Prerequisites

- Python 3.10 or newer.
- `numpy` and `keras` 3.x with a backend installed:
  ```bash
  pip install numpy keras tensorflow-cpu
  ```
  Any Keras 3 backend works. `tensorflow-cpu` is the smallest of the three that
  installs cleanly on most machines.
- Internet access on first run. Keras downloads MNIST (11490434 bytes) to
  `~/.keras/datasets/` and caches it there.
- Almost no CPU time. 15 epochs over 54000 images on a 7960-parameter model
  measures about 6 seconds on a laptop CPU, and the integer evaluation that
  follows takes another 9. No GPU needed.

## Steps

1. **Back up the shipped weights.** `train_new_model` overwrites its target
   without prompting, and the repository's trained model is the only copy.

   ```bash
   cd neuralnetwork/MNSIT-NEURAL-NETWORK
   cp mnist_model.keras mnist_model.keras.bak
   ```

2. **Enable the training call.** In `main()`, uncomment the training line:

   ```python
   def main():
       model_name = 'mnist_model.keras'
       train_new_model(model_name)     # <- uncomment this
       forward(model_name, 10000)
   ```

   Leaving `forward` in place means the script trains and then immediately
   evaluates the new weights through the integer pass, which is exactly the
   check you want.

3. **Run it.**

   ```bash
   python main.py
   ```

   Keras prints a progress bar per epoch:

   ```
   Epoch 1/15
   422/422 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.7... - loss: 0.9... - val_accuracy: 0.8... - val_loss: 0.5...
   ```

   422 steps is 54000 training images at batch size 128, after the 10 percent
   validation split.

4. **Read the float metrics.** After the last epoch, `train_new_model` saves and
   evaluates. A measured retraining run produced:

   ```
   Test Loss 0.27068477869033813
   Test Accuracy 0.9229999780654907
   ```

   This is the ceiling. It is the Keras model in float32, before any
   quantization. Write the number down. The shipped model scores 0.9237, so a
   fresh run landing within a few tenths of a point is normal.

5. **Let the integer pass run.** `forward` then loads the model you just saved,
   quantizes it, and classifies the test set, printing a running count every 100
   images and a final line:

   ```
   9900
   Accuracy: 8544 / 10000 = 85.44%
   ```

   That figure is the shipped model's. The whole pass takes about 9 seconds, so
   there is no need to sample; run all 10000.

6. **Re-comment the training call** once you have weights you want to keep, so a
   later run does not silently overwrite them:

   ```python
   # train_new_model(model_name)
   ```

## Verification

Three things should hold:

- `mnist_model.keras` has a fresh modification time.
- The float test accuracy from step 4 is around 0.92. A 10-unit hidden layer on
  binarized MNIST is a small model; anything below 0.5 means training diverged.
- The integer accuracy from step 5 is **lower than** the float accuracy but not
  catastrophically so. The expected gap is just under 7 points: 92.37% float
  against 85.44% integer on the shipped weights. A collapse to roughly 10
  percent means the integer pass is broken, not merely lossy, and accumulator
  overflow is the usual cause. See
  [How to tune the fixed-point format](howto-tune-fixed-point.md).

Confirm the saved architecture directly if you want certainty:

```bash
python -c "import zipfile,json; print(json.loads(zipfile.ZipFile('mnist_model.keras').read('config.json'))['config']['layers'])"
```

You should see `InputLayer`, `Dense`, `Activation`, `Dense`, `Activation` in
that order.

## Troubleshooting

**`ModuleNotFoundError: No module named 'keras'`**
Keras is not installed, or you are in the wrong interpreter. Install it with the
command in the prerequisites, and check `python -c "import sys; print(sys.executable)"`
points where you expect.

**`ValueError: Invalid filepath extension for saving`**
Keras 3 only saves to `.keras` (native) or `.h5` (legacy). Change `model_name`
in `main()` to end in `.keras`. Verified: `.model` and `.pkl` both raise.

**`ValueError: cannot reshape array of size N into shape (60000,784)`**
`clean_data` hardcodes 60000 training and 10000 test images. It only accepts the
standard MNIST split. To use a different dataset, replace the literals with
`X_train.shape[0]` and `X_test.shape[0]`.

**`IndexError: list index out of range` inside `forward`**
`forward` reads `model.layers[0]` and `model.layers[2]` as the two `Dense`
layers, which only holds while the activations are separate `Activation` layers.
If you rewrote the model as `Dense(10, activation='relu')`, the Dense layers are
now at indices `0` and `1`. Either restore the separate `Activation` layers or
update the indices in `forward`.

**Training runs but accuracy hovers near 0.10**
Ten percent is chance on 10 classes. Check that `clean_data` still returns
one-hot labels of width 10 and that the binarization threshold did not push
every pixel to the same value. Print `X_train.sum()` to confirm some pixels
survive.

**MNIST download hangs or fails**
Keras fetches from a Google Storage URL on first use. Behind a proxy, set
`HTTPS_PROXY`, or download `mnist.npz` manually into `~/.keras/datasets/`.

**Overflow warnings during the `forward` phase**
The shipped weights produce zero overflow events at the default 2 fractional
bits, so any warning here means your retrained weights are larger than the
originals. This is the `int8` accumulator wrapping, which is faithful to the
hardware, but it will cost accuracy. See
[Explanation: accumulator width is the whole game](explanation-integer-inference.md#accumulator-width-is-the-whole-game).

## Related

- [Reference: `train_new_model`](reference-api.md#train_new_modelmodel_name)
- [How to tune the fixed-point format](howto-tune-fixed-point.md)
- [Tutorial: your first run](tutorial-first-run.md)
- [Explanation: why this network runs on integers](explanation-integer-inference.md)
