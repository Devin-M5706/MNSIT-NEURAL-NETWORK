"""MNIST classifier trained in float, then re-run in 8/16-bit integer arithmetic.

The integer path in forward() exists so the same computation can be rebuilt in
Minecraft redstone: adders, comparators, and gates only. No floating point, no
exponentials, no division at inference.

Several things below look like bugs and are load-bearing. Before changing them,
read docs/explanation-integer-inference.md:

  * The accumulators are NumPy int8/int16 and WRAP on overflow. That models an
    8-bit adder with no carry-out. Widening them makes the reported accuracy
    unreachable by the hardware.
  * check_overflow() can never fire as called, because its argument has already
    wrapped into range. See docs/howto-tune-fixed-point.md for the fix.
  * The activations are separate Activation layers, not Dense(activation=...),
    because forward() addresses the Dense layers as layers[0] and layers[2].
"""
import numpy as np
import keras
from keras import layers

def clean_data(data):
    """Flatten, normalize, and BINARIZE MNIST to one bit per pixel.

    Binarization is the highest-leverage decision in the project: when every
    input is 0 or 1, `weight * pixel` collapses to `add the weight or skip it`,
    so the 784-wide layer needs zero multipliers.

    Returns X as int8 in {0, 1} and Y as float64 one-hot, both train and test.
    Sizes are hardcoded, so only the standard 60000/10000 MNIST split works.
    """
    (X_train, Y_train), (X_test, Y_test) = data

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = np.where(X_train >= 0.5, 1, 0).astype('int8')  # (60000 x 784) binary values
    X_test = np.where(X_test >= 0.5, 1, 0).astype('int8')  # (10000 x 784) binary values

    Y_train = keras.utils.to_categorical(Y_train, 10)  # (60000 x 10) 1-hot encoded
    Y_test = keras.utils.to_categorical(Y_test, 10)  # (10000 x 10) 1-hot encoded

    return X_train, Y_train, X_test, Y_test

def train_new_model(model_name):
    """Train on binarized MNIST and overwrite model_name. Takes ~6s on a CPU.

    Trains in float32; forward() quantizes afterwards. ReLU rather than sigmoid
    because max(0, x) is one sign-bit test in redstone while e^-x needs a
    lookup table. Softmax is here only so crossentropy has a distribution to
    fit; forward() drops it, which is exact because softmax is monotonic.
    """
    X_train, Y_train, X_test, Y_test = clean_data(keras.datasets.mnist.load_data())

    model = keras.Sequential(
        [
            keras.Input(shape=(784,)),
            layers.Dense(10),
            # Separate Activation layers, NOT Dense(10, activation='relu').
            # forward() reads layers[0] and layers[2]; collapsing these shifts
            # the Dense layers to indices 0 and 1 and breaks it.
            layers.Activation('relu'),
            layers.Dense(10),
            layers.Activation('softmax')
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=128, epochs=15, validation_split=0.1)
    model.save(model_name)

    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
    print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])

def check_overflow(x, num_bits):
    """Warn if x falls outside a signed num_bits register.

    KNOWN LIMITATION: both call sites in forward() pass a NumPy scalar that has
    ALREADY wrapped to the target width, so x is inside the tested range by
    construction and this never prints. Real overflow surfaces as NumPy's
    "RuntimeWarning: overflow encountered in scalar add" instead. To actually
    catch it, wrap the loop in np.errstate(over='raise') or accumulate in int32
    first; see docs/howto-tune-fixed-point.md.
    """
    if (x > (2 ** (num_bits-1) - 1)) or (x < -(2 ** (num_bits-1))):
        print('overflow detected')

def forward(model_name, iterations=5000):
    """Classify test images using only integer add, compare, and max.

    This is the function the redstone build has to reproduce. Prints a running
    count every 100 images and a final "Accuracy: a / b = c%" line. iterations
    is capped by the 10000-image test set. model_name resolves against the
    current working directory.

    Measured on the shipped weights: 8544 / 10000 = 85.44%, about 9s.
    """
    X_train, Y_train, X_test, Y_test = clean_data(keras.datasets.mnist.load_data())

    model = keras.saving.load_model(model_name)
    # layers[0] and layers[2] are the Dense layers; 1 and 3 are the Activations.
    # get_weights() returns exactly [kernel, bias] per Dense layer.
    weights1 = model.layers[0].get_weights()[0]
    biases1 = model.layers[0].get_weights()[1]
    weights2 = model.layers[2].get_weights()[0]
    biases2 = model.layers[2].get_weights()[1]

    def to_fixed(float_value, bits_past_radix=2):
        """Float to fixed-point int: store round(v * 4), meaning steps of 0.25.

        Fixed point makes fractional weights addable by a plain binary adder,
        which is the most basic redstone arithmetic circuit there is.

        Raising bits_past_radix improves weight resolution and doubles every
        partial sum. Measured on the shipped model: 2 bits scores 85.44% with
        zero overflow, 3 scores 86.67% with 1370, 4 collapses to 28.65%.
        """
        a = float_value * (2 ** bits_past_radix)
        b = int(round(a))
        if a < 0:
            # Two's-complement negation spelled out to document the circuit
            # operation. In Python this returns the value b already holds.
            b = ~(abs(b)) + 1
        return b

    weights1 = np.vectorize(to_fixed)(weights1).astype('int8')
    biases1 = np.vectorize(to_fixed)(biases1).astype('int8')
    weights2 = np.vectorize(to_fixed)(weights2).astype('int8')
    biases2 = np.vectorize(to_fixed)(biases2).astype('int8')

    count = 0
    total = 0

    for X, Y in zip(X_test, Y_test):
        # HIDDEN LAYER

        output = [0] * 10
        for neuron in range(10):
            weights = weights1.T[neuron]

            # `weight` starts as a Python int and becomes np.int8 on the first
            # +=, so it WRAPS at 127 rather than promoting. That is deliberate:
            # it is what an 8-bit redstone adder does. Only ~13% of pixels are
            # lit, so the `if` skips most of these 784 iterations.
            weight = 0
            for index, pixel in enumerate(X):
                if pixel == 1:
                    # No multiply: a binary pixel means add the weight or skip.
                    weight += weights[index]
                    check_overflow(weight, 8)

            weight += biases1.T[neuron]
            check_overflow(weight, 8)

            output[neuron] = int(weight)

        hidden_out = np.array(output).astype('int8')

        # RELU
        # In hardware this is one sign-bit test: if the top bit is set, output
        # zero, otherwise pass the value through. One inverter and eight gates.

        hidden_out = np.maximum(0, hidden_out).astype('int8')

        # OUTPUT LAYER
        # The only real multipliers in the network live here: 10 per neuron,
        # 100 total. Both operands vary, so there is no way to gate them away
        # the way the hidden layer does. int8 * int16 promotes to int16, which
        # is the headroom that keeps products of two int8 values from wrapping.

        output = [0] * 10
        for neuron in range(10):
            weights = weights2.T[neuron]

            weight = 0
            for index, value in enumerate(hidden_out):
                weight += weights[index] * np.int16(value)
                check_overflow(weight, 16)

            weight += biases2.T[neuron]
            check_overflow(weight, 16)

            output[neuron] = int(weight)

        output_out = np.array(output).astype('int16')
        # Argmax as a one-hot vector. No softmax: it is monotonic, so it cannot
        # change which output is largest, and skipping it removes ten
        # exponentials and a division from the circuit.
        # EDGE CASE: ties light every maximum, so the result cannot equal a
        # one-hot label and the image scores wrong even if one tied neuron was
        # right. 56 of 10000 images hit this at the default quantization.
        prediction = np.where(output_out == np.max(output_out), 1, 0).astype('int16')

        if np.array_equal(prediction, Y):
            count += 1
        total+=1

        # Break precedes the progress print, so the final multiple of 100 is
        # never shown: a 10000-image run emits 99 progress lines, not 100.
        if total >= iterations:
            break

        if total % 100 == 0:
            print(total)


    print(f'Accuracy: {count} / {total} = {count / total * 100}%')

def main():
    model_name = 'mnist_model.keras'
    # train_new_model(model_name)
    forward(model_name, 10000)

if __name__ == '__main__':
    main()