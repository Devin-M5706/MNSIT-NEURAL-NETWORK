# How to tune the fixed-point format

Change how many fractional bits the quantized weights carry, detect the overflow
that change causes, and measure what it costs in accuracy. The end result is a
number you can defend when you pick a wire width for the redstone build.

No retraining is involved. Quantization happens inside `forward()` at load time,
so every experiment here runs against the weights already in
`mnist_model.keras`, and a full 10000-image evaluation takes about 9 seconds.

## Prerequisites

- A working run of the simulator. If you have not done one, start with
  [Tutorial: your first run](tutorial-first-run.md).
- `numpy` 2.x and `keras` 3.x with a backend.
- Familiarity with what the format means:
  [Explanation: fixed point instead of floating point](explanation-integer-inference.md#3-fixed-point-instead-of-floating-point).

Work in `neuralnetwork/MNSIT-NEURAL-NETWORK/`. Every path below is relative to
it.

## The knob

Inside `forward()`:

```python
def to_fixed(float_value, bits_past_radix=2):
    a = float_value * (2 ** bits_past_radix)
    b = int(round(a))
    ...
```

`bits_past_radix` sets the scale. Two bits means multiply by 4, so one integer
step represents 0.25 and an `int8` covers `-32.0` to `+31.75`.

| `bits_past_radix` | Scale | Resolution | `int8` weight range |
|---|---|---|---|
| 1 | 2 | 0.5 | -64.0 to +63.5 |
| 2 (default) | 4 | 0.25 | -32.0 to +31.75 |
| 3 | 8 | 0.125 | -16.0 to +15.875 |
| 4 | 16 | 0.0625 | -8.0 to +7.9375 |

Every bit you add halves the rounding error on each weight and doubles the
magnitude of everything the accumulator sums. That is the entire trade.

## What the trade actually costs

Measured against the shipped `mnist_model.keras`, full 10000-image test set:

| Bits | Accuracy | Overflow events | Ties | Saturating weights |
|---|---|---|---|---|
| 0 | 9.20%\* | 0 | 44\* | 0 |
| 1 | 42.00%\* | 0 | 68\* | 0 |
| 2 (default) | 85.44% | 0 | 56 | 0 |
| 3 | **86.67%** | 1370 | 10 | 0 |
| 4 | 28.65% | 46538 | 10 | 0 |
| 5 | 8.35%\* | 27007\* | 3\* | 0 |
| 6 | 9.35%\* | 59546\* | 0\* | 0 |
| 7 | 9.85%\* | 122799\* | 0\* | 6 |

\* measured over the first 2000 images rather than all 10000. Bits 2 through 4
are full-test-set figures.

For reference, the float32 Keras model scores **92.37%** on the same test set.
That is the ceiling.

Read three things off this table before you start turning the knob:

- **The default is not the optimum.** 3 bits beats 2 by 1.2 points. The default
  is the last setting that overflows zero times, which is a reasonable thing to
  optimize for in hardware, but it is not the accuracy peak.
- **Above 3 bits it is a cliff, not a slope.** 86.67% to 28.65% in one step. A
  wrapped accumulator flips a large positive sum negative, ReLU clamps it to
  zero, and the neuron goes dark. There is no graceful degradation to ride.
- **Weight saturation is a red herring for this model.** The largest absolute
  weight is 1.6448, so nothing saturates until 7 bits, long after overflow has
  destroyed the result. Check it anyway if you retrain, because a different
  training run can produce larger weights.

## Steps

1. **Change the default.** Edit the nested `to_fixed` signature:

   ```python
   def to_fixed(float_value, bits_past_radix=3):
   ```

   Alternatively, edit the four `np.vectorize(to_fixed)` call sites if you want
   different precision per layer. The output layer accumulates in `int16` and
   tolerates more bits than the hidden layer does.

2. **Check weight saturation.** Cheap, and it catches the failure mode that
   looks identical to overflow from the outside:

   ```bash
   python -c "
   import keras, numpy as np
   m = keras.saving.load_model('mnist_model.keras')
   for name, idx in (('layer1', 0), ('layer2', 2)):
       w = m.layers[idx].get_weights()[0]
       for bits in (1, 2, 3, 4, 5, 6, 7):
           scaled = np.round(w * (2 ** bits))
           bad = np.sum((scaled > 127) | (scaled < -128))
           print(f'{name} bits={bits}: {bad} of {w.size} weights saturate')
   "
   ```

   On the shipped model every line reads `0` until `bits=7`. Any nonzero count
   means those weights are being corrupted rather than rounded: `.astype('int8')`
   wraps them, so a weight of +40.0 stores as 64 and reads back as +8.0.

3. **Run an evaluation.** The full test set only takes about 9 seconds, so there
   is no reason to sample:

   ```python
   def main():
       model_name = 'mnist_model.keras'
       # train_new_model(model_name)
       forward(model_name, 10000)
   ```

   ```bash
   python main.py
   ```

4. **Record the final line.**

   ```
   Accuracy: 8667 / 10000 = 86.67%
   ```

   If you do evaluate on a subset instead, compare like with like. The first 200
   test images score 90.0% at the default while the full set scores 85.44%, so
   subset results are not comparable across different sizes.

5. **Count the overflow you just bought.** Accuracy alone will not tell you
   whether you are near the cliff edge. See the next section.

## Making overflow detection actually work

`check_overflow` is called in both accumulation loops and cannot ever fire as
written. The accumulator is a NumPy `int8`, so it has already wrapped into range
by the time it is passed in:

```python
>>> np.int8(100) + np.int8(100)
np.int8(-56)      # RuntimeWarning: overflow encountered in scalar add
```

`-56` passes an 8-bit range test. The information is gone. At the default of 2
bits this hides nothing, because the true count is zero. At 3 bits it hides 1370
events.

There are two fixes, and they answer different questions.

### Fix A: turn NumPy's warning into an exception

Fastest way to find out *whether* overflow happens at all. Wrap the per-image
loop in `forward()`:

```python
import numpy as np

with np.errstate(over='raise'):
    ...   # the existing per-image loop
```

Overflow now raises `FloatingPointError: overflow encountered in scalar add` at
the exact iteration. Verified on NumPy 2.2 and 2.5.

To count rather than stop, use the warnings module:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    ...   # the existing per-image loop
print(f'{len(caught)} overflow events')
```

This is how the table above was produced.

### Fix B: accumulate wide, then check

Answers *by how much* it overflows, which is what tells you the wire width you
actually need. Widen the accumulator so it cannot wrap, and let `check_overflow`
do the job it was written for:

```python
weight = np.int32(0)
for index, pixel in enumerate(X):
    if pixel == 1:
        weight += np.int32(weights[index])
check_overflow(weight, 8)          # now this can genuinely fire
```

`int32` holds 784 additions of `int8` values with room to spare, so the true sum
survives and the 8-bit range test becomes meaningful.

**This changes what the simulation simulates.** With Fix B the script computes
the mathematically correct sum, which an 8-bit redstone adder will not. Use it
to size your hardware, then revert to the wrapping `int8` version to model what
the hardware will really do. See
[Explanation: accumulator width is the whole game](explanation-integer-inference.md#accumulator-width-is-the-whole-game).

## Verification

You have tuned this correctly when all three hold:

- **No saturating weights** at your chosen bit count. Step 2 reports 0.
- **An overflow count you have consciously accepted.** Zero if you want the
  circuit to be provably correct at 8 bits; nonzero only if you have measured
  that the accuracy is still better despite it, as it is at 3 bits.
- **Integer accuracy within a few points of 92.37%.** A gap of 7 points is the
  normal quantization cost here. A drop toward 10 percent means the accumulator
  is wrapping constantly, not that quantization is merely lossy.

Sanity-check the round trip on a single weight at any time:

```bash
python -c "
import keras, numpy as np
m = keras.saving.load_model('mnist_model.keras')
w = m.layers[0].get_weights()[0][0][0]
for bits in (1, 2, 3, 4):
    q = int(round(w * 2**bits))
    print(f'bits={bits}: {w:+.5f} -> stored {q:4d} -> reads back {q / 2**bits:+.5f}')
"
```

The reconstruction error should halve with each added bit.

## Troubleshooting

**Accuracy collapses to about 10 percent**
Ten percent is chance on 10 classes. At 4 or more fractional bits this is
expected and is caused by accumulator overflow, not by anything you did wrong.
Confirm with Fix A and step back down to 2 or 3.

**Accuracy does not change when you change the bit count**
Confirm you edited the `bits_past_radix` default and not a copy, and that
`forward` is the function being called. Quantization is applied by four
`np.vectorize(to_fixed)` calls immediately after the weights are loaded.

**Overflow warnings appear only on some images**
Normal. Whether a neuron overflows depends on how many pixels are lit, so dense
digits like 8 trigger it and sparse ones like 1 do not.

**`RuntimeWarning` lines flood the output**
NumPy prints each scalar overflow once per source location by default. Use the
`catch_warnings(record=True)` form above to collect them silently and print a
single count.

**Fix B reports no overflow but accuracy is still poor**
Then overflow is not your problem. Check weight saturation in step 2, and the
tie-handling edge case in
[Reference: known edge cases](reference-api.md#known-edge-cases). Ties rise
sharply at coarse quantization: 68 per 2000 images at 1 bit, against 10 at 3
bits.

## Related

- [Reference: `forward` and `to_fixed`](reference-api.md#forwardmodel_name-iterations5000)
- [Reference: `check_overflow`](reference-api.md#check_overflowx-num_bits)
- [Explanation: the 8-bit ceiling, measured](explanation-integer-inference.md#the-8-bit-ceiling-measured)
- [Explanation: mapping the network to redstone](explanation-redstone-mapping.md)
- [How to retrain the model](howto-retrain-the-model.md)
