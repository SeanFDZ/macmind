# MacMind

**A complete transformer neural network implemented entirely in HyperTalk,  trained on a Macintosh SE/30.**

MacMind is a 1,216-parameter single-layer single-head transformer that learns the bit-reversal permutation -- the opening step of the Fast Fourier Transform -- from random examples.  Every line of the neural network is written in HyperTalk,  a scripting language from 1987 designed for making interactive card stacks,  not matrix math.  It has token embeddings,  positional encoding,  self-attention with scaled dot-product scores,  cross-entropy loss,  full backpropagation,  and stochastic gradient descent.  No compiled code.  No external libraries.  No black boxes.

Option-click any button and read the actual math.

<!-- TODO: Screenshot of Card 2 (Training) mid-training, showing progress bars and position accuracy grid -->

---

## Why This Exists

The same fundamental process that trained MacMind -- forward pass,  loss computation,  backward pass,  weight update,  repeat -- is what trained every large language model that exists today.  The difference is scale,  not kind.  MacMind has 1,216 parameters.  GPT-4 has roughly a trillion.  The math is identical.

We are at a moment where AI affects nearly everyone but almost nobody understands what it actually does.  MacMind is a demonstration that the process is knowable -- that backpropagation and attention are not magic,  they are math,  and that math does not care whether it is running on a TPU cluster or a 68000 processor from 1987.

Everything is inspectable.  Everything is modifiable.  Change the learning rate,  swap the training task,  resize the model -- all from within HyperCard's script editor.  This is the engine with the hood up.

---

## What It Learns

The bit-reversal permutation reorders a sequence by reversing the binary representation of each position index.  For an 8-element sequence:

```
Position:    0    1    2    3    4    5    6    7
Binary:     000  001  010  011  100  101  110  111
Reversed:   000  100  010  110  001  101  011  111
Maps to:     0    4    2    6    1    5    3    7
```

So input `[3, 7, 1, 9, 5, 2, 8, 4]` becomes `[3, 5, 1, 8, 7, 2, 9, 4]`.

This permutation is the first step of the Fast Fourier Transform,  one of the most important algorithms in computing.  The model is never told the rule.  It discovers the positional pattern purely through self-attention and gradient descent -- the same process,  scaled up enormously,  that taught larger models to understand language.

After training,  the attention map on Card 4 reveals the butterfly routing pattern of the FFT.  The model independently discovered the same mathematical structure that Cooley and Tukey published in 1965.

<!-- TODO: Screenshot of Card 4 (Attention Map) after training, showing the butterfly pattern -->

---

## The Stack

MacMind is a 5-card HyperCard stack:

| Card | Purpose |
|------|---------|
| **1 -- Title** | Project name and credits |
| **2 -- Training** | Train the model and watch it learn in real time |
| **3 -- Inference** | Test the trained model on any 8-digit input |
| **4 -- Attention Map** | Visualize the 8x8 attention weight matrix |
| **5 -- About** | Plain-text explanation of what the model is doing |

<!-- TODO: Screenshot of Card 3 (Inference) showing a correct prediction with confidence scores -->

### Training (Card 2)

Click **Train 10** for 10 training steps,  or **Train to 100%** to train until the model gets a perfect score on a sample.  For deeper training,  run **Train 10** repeatedly or click **Train to 100%** again -- the model picks up where it left off.  For a longer run,  open the Message Box (Cmd-M) and type `trainN 1000` to train for 1,000 steps straight.

Each step generates a random 8-digit sequence,  runs the full forward pass,  computes cross-entropy loss,  backpropagates gradients through every layer,  and updates all 1,216 weights.  Progress bars,  per-position accuracy,  and a training log update in real time.

### Inference (Card 3)

After training,  click **New Random** to generate a test input,  then **Permute** to run the trained model.  The output row shows the model's predictions and the confidence row shows how sure it is about each position.

To verify the result,  apply the bit-reversal permutation by hand.  The output should rearrange the input positions in this order:

```
Output[0] = Input[0]        Output[4] = Input[1]
Output[1] = Input[4]        Output[5] = Input[5]
Output[2] = Input[2]        Output[6] = Input[3]
Output[3] = Input[6]        Output[7] = Input[7]
```

For example,  input `[3, 7, 1, 9, 5, 2, 8, 4]` should produce `[3, 5, 1, 8, 7, 2, 9, 4]`.  If the model is well-trained,  every position will be correct with confidence above 90%.

### Attention Map (Card 4)

The 8x8 grid visualizes which input positions the model attends to when producing each output position.  After training,  you should see the butterfly pattern: positions 0, 2, 5, 7 attend to themselves (fixed points of the permutation),  while positions 1 and 4 attend to each other,  and positions 3 and 6 attend to each other (swap pairs).

This is the same routing structure discovered by Cooley and Tukey in 1965 for the Fast Fourier Transform:

![FFT Butterfly Diagram](https://upload.wikimedia.org/wikipedia/commons/9/98/Butterfly-FFT.png)
*The classic FFT butterfly diagram ([public domain](https://en.wikipedia.org/wiki/Butterfly_diagram)).  The model discovers this structure independently through attention.*

<!-- TODO: Screenshot of Card 4 (Attention Map) after training, showing the butterfly pattern -->

---

## Architecture

| Component | Dimensions | Parameters |
|-----------|-----------|------------|
| Token embeddings (W_embed) | 10 x 16 | 160 |
| Position embeddings (W_pos) | 8 x 16 | 128 |
| Query projection (W_Q) | 16 x 16 | 256 |
| Key projection (W_K) | 16 x 16 | 256 |
| Value projection (W_V) | 16 x 16 | 256 |
| Output projection (W_out) | 16 x 10 | 160 |
| **Total** | | **1,216** |

Data flow:

```
Input digits [8]
    |
Token embedding lookup + position embedding --> [8 x 16]
    |
Q, K, V projections --> [8 x 16] each
    |
Attention scores = Q x K^T, scaled by 1/sqrt(16) --> [8 x 8]
    | softmax per row
Attention weights --> [8 x 8]
    |
Context = weights x V --> [8 x 16]
    |
Residual connection: context + embedded input --> [8 x 16]
    |
Output logits = residual x W_out --> [8 x 10]
    | softmax per position
Predictions --> [8 x 10] probability distribution over digits
```

All weights and activations are stored as comma-delimited numbers in hidden HyperCard fields on Card 2.  A 16x16 weight matrix is 256 comma-separated values in a single field.  Save the stack,  quit,  reopen it: the trained model is still there.

---

## Training on Real Hardware

MacMind was trained on a Macintosh SE/30 running System 7.6.1 and has also been tested through Basilisk II on Apple Silicon.  HyperTalk is interpreted,  and every multiply,  every field access,  every variable lookup goes through the interpreter.  Each training step takes several seconds.  Training to convergence (~1,000 steps) takes hours.

The model was left training overnight,  grinding through backpropagation one 8 MHz multiply-accumulate at a time.  By morning it had learned the permutation.

---

## Requirements

**HyperCard 2.0 or later** is required.  HyperCard 1.x evaluates arithmetic left-to-right without standard precedence (`2 + 3 * 4 = 20` instead of `14`),  which would silently corrupt every matrix multiplication and gradient computation in the model.  HyperCard 2.0 introduced standard mathematical operator precedence.  The stack was built and tested with HyperCard 2.1.

| | HyperCard 2.1 Minimum | MacMind Reference |
|---|---|---|
| **HyperCard** | 2.0 | 2.1 |
| **System software** | System 6.0.5 | System 7.6.1 |
| **RAM** | 1 MB (2 MB recommended) | 4 MB |
| **Processor** | 68000 | 68030 (Mac SE/30) |
| **Also runs on** | Mac OS 8,  Mac OS 9,  Mac OS X Classic Environment (through 10.4 Tiger on PowerPC) | |

On real vintage hardware,  each training step takes several seconds and full training takes hours.  On a modern Mac running Basilisk II or SheepShaver,  performance is comparable -- HyperTalk interpretation is the bottleneck,  not the host CPU.

---

## Running It Yourself

### Quick Start (pre-trained)

1. Download `MacMind-Trained.dsk` from [Releases](https://github.com/SeanFDZ/macmind/releases)
2. Open it in SheepShaver,  Basilisk II,  or Mini vMac running System 7.x through Mac OS 9
3. Double-click the MacMind stack
4. Navigate to Card 3 (Inference),  click **New Random**,  then **Permute**

### Watch It Learn (blank stack)

1. Download `MacMind-Blank.dsk` from [Releases](https://github.com/SeanFDZ/macmind/releases)
2. Open it in your emulator
3. Navigate to Card 2 (Training)
4. Click **Train 10** for short runs,  or **Train to 100%** to train until the model gets a perfect score on a sample.  For a longer run,  open the Message Box (Cmd-M) and type `trainN 1000` to train for 1,000 steps straight.  The model picks up where it left off each time.

### Validate the Math (Python)

The `validate.py` script is a Python/NumPy reference implementation of the exact same transformer.  It trains on the same task with the same architecture and confirms convergence to 100% accuracy.

```
python3 validate.py
```

---

## Credits

- **[Frank Rosenblatt (1958)](https://en.wikipedia.org/wiki/Perceptron)** -- the Perceptron: first demonstration that a machine can learn from examples by adjusting weights
- **[Paul Werbos (1974) / Rumelhart,  Hinton & Williams (1986)](https://en.wikipedia.org/wiki/Backpropagation)** -- backpropagation: the training algorithm this project implements
- **[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)** -- "Attention Is All You Need": the transformer architecture this model implements
- **[Cooley & Tukey (1965)](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)** -- the Fast Fourier Transform algorithm whose bit-reversal permutation is the training task

MacMind is an original implementation by [Sean Lavigne](https://fallingdata.zone).

---

## License

MIT.  See [LICENSE](LICENSE).
