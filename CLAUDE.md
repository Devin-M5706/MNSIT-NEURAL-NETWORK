# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A neural network trained on MNIST whose **forward pass is re-implemented in 8-bit
and 16-bit integer arithmetic**, so it can eventually be rebuilt in Minecraft
redstone. The constraint driving every design decision is that redstone has
adders and comparators but no floating point unit, no cheap multiplier, and no
`e^x`.

Read `docs/explanation-integer-inference.md` before changing anything in
`forward()`. Several things that look like bugs are load-bearing (see
"Do not 'fix' these" below).

## Commands

All code lives in `neuralnetwork/MNSIT-NEURAL-NETWORK/`. **You must run from that
directory** because `main.py` loads `'mnist_model.keras'` by relative path.

```bash
pip install -r requirements.txt            # numpy, keras, tensorflow-cpu, pytest
cd neuralnetwork/MNSIT-NEURAL-NETWORK
python main.py                             # integer eval over 10000 test images, ~9s
```

To retrain, uncomment `train_new_model(model_name)` in `main()` (~6s, CPU only).
It **overwrites `mnist_model.keras` without prompting**; back it up first.

To evaluate fewer images, change the second argument: `forward(model_name, 500)`.
There is rarely a reason to; the full pass is 9 seconds.

**Tests** run from the repository root, not the code directory:

```bash
pytest tests/                     # 35 tests, ~29s
pytest tests/ -m "not slow"       # skip full-test-set evaluation, ~14s
pytest tests/test_forward.py::TestAccuracy -v          # one class
pytest tests/test_data.py::test_shapes                 # one test
```

Tests skip with a message when Keras or `mnist_model.keras` is missing. There is
no linter config and no CI. `tests/test_forward.py::TestAccuracy::test_full_test_set_regression_lock`
asserts the exact figure `8544 / 10000`; the forward pass has no RNG, so that is
a deterministic lock, not a flaky threshold. Retraining moves it and the docs
need re-measuring.

## Measured baselines

Regression checks. All from the shipped `mnist_model.keras` over the full 10000
test images:

| | |
|---|---|
| Float32 Keras model | 92.37% |
| Integer forward pass (default 2 fractional bits) | **85.44%** (`Accuracy: 8544 / 10000 = 85.44%`) |
| Overflow events at 2 bits | 0 |
| Ties at 2 bits | 56 |
| Full run wall clock | ~9s |
| Retrain wall clock | ~6s |

`bits_past_radix` sweep (the `to_fixed` default inside `forward`):

| Bits | Accuracy | Overflow events |
|---|---|---|
| 2 (default) | 85.44% | 0 |
| 3 | 86.67% | 1370 |
| 4 | 28.65% | 46538 |

The default is **not** the accuracy optimum. It is the last setting that
overflows zero times, which is the right thing to optimize for a physical
circuit. Do not "improve" it to 3 without saying why.

## Architecture

```
Input 784 (binarized to 1 bit)  ->  Dense 10  ->  ReLU  ->  Dense 10  ->  softmax
```

7960 parameters. Training happens in float via Keras; **inference is
re-implemented by hand in `forward()`** using only integer add, compare, and
`max`. The two paths are independent, and `forward()` is the one that matters.

Three choices make the network buildable, and they are why the architecture
looks undersized for MNIST:

1. **Binary input.** `clean_data` thresholds pixels at 0.5, so a weight is either
   added or skipped. The 784-wide layer needs **zero multipliers**. This is the
   single highest-leverage decision in the project.
2. **ReLU, not sigmoid.** `max(0, x)` is one sign-bit test. Sigmoid would need a
   lookup table for `e^-x`.
3. **No softmax at inference.** Softmax is monotonic, so it cannot change the
   argmax. Skipping it is exact, not an approximation.

## Do not "fix" these

Each looks like a defect and is intentional. Changing any of them silently
breaks the point of the project.

- **`int8` / `int16` accumulators wrap on overflow.** In `forward()`, `weight = 0`
  becomes a NumPy `int8` on its first `+=` and wraps rather than promoting. That
  models an 8-bit redstone adder. Widening it to `int32` makes the script report
  accuracy the hardware cannot reproduce.
- **`check_overflow` can never fire.** Its callers pass an already-wrapped scalar,
  so the value is inside the tested range by construction. Overflow surfaces as
  NumPy's `RuntimeWarning` instead. Fixes exist
  (`docs/howto-tune-fixed-point.md#making-overflow-detection-actually-work`) but
  they change what is being simulated; use them as a measurement tool, not a
  permanent edit.
- **`to_fixed`'s `b = ~(abs(b)) + 1` branch is a no-op.** It spells out
  two's-complement negation to document the circuit operation. It is a comment
  written as code.
- **Layer indices 0 and 2 are load-bearing.** `forward()` reads
  `model.layers[0]` and `model.layers[2]` as the two `Dense` layers. That only
  holds while activations are separate `Activation` layers. Collapsing them into
  `Dense(10, activation='relu')` shifts the indices to 0 and 1 and breaks
  `forward` with an `IndexError`.
- **Root `main.py` is `print("hello world")`.** Vestigial, not an entry point.
  All real code is in `neuralnetwork/MNSIT-NEURAL-NETWORK/main.py`.

## Known rough edges

Real limitations, safe to fix if asked:

- `clean_data` hardcodes 60000/10000 reshape sizes, so it only accepts the
  standard MNIST split.
- Argmax is written as `np.where(out == np.max(out), 1, 0)`, so **ties light
  multiple outputs** and always score as wrong. Integer outputs tie far more
  often than floats: 56 times per 10000 images at the default.
- Progress printing stops at 9900 on a 10000-image run; the loop breaks before
  its final print. All images are still classified.

## Documentation

`docs/` follows Diataxis. Keep new docs in the matching quadrant and cross-link
them; `README.md` is the index and every doc must stay reachable from it.

| File | Quadrant |
|---|---|
| `docs/tutorial-first-run.md` | Tutorial |
| `docs/howto-retrain-the-model.md`, `docs/howto-tune-fixed-point.md` | How-to |
| `docs/reference-api.md` | Reference |
| `docs/explanation-integer-inference.md`, `docs/explanation-redstone-mapping.md` | Explanation |

Diagrams live in `diagrams/` as mermaid sources plus rendered SVG/PNG and
editable `.excalidraw` scenes. The two explanation docs embed the same mermaid
in ` ```mermaid ` fences, which GitHub renders natively. **Edit the `.mmd` and
the fence together** so they cannot drift.

Figures in these docs are measured, not estimated. If you change behavior,
re-measure and update the numbers rather than hedging them.

`docs/explanation-redstone-mapping.md` describes a **design target**. No
schematic, world save, or command-block generator exists in this repository.
Keep that distinction explicit in anything you write about the build.
