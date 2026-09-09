# FlowTransformer for the Harmonic Oscillator

Consider the three-dimensional harmonic oscillator

```math
q''(t) = -q(t),
\qquad
q(t) \in \mathbb{R}^3.
```

Writing

```math
v = q',
```

the corresponding first-order system is

```math
q' = v,
\qquad
v' = -q.
```

The state of the system is

```math
z = (q,v) \in \mathbb{R}^6.
```

The exact flow map is

```math
\Psi(t,q,v)
=
\left(
q\cos(t) + v\sin(t),
-q\sin(t) + v\cos(t)
\right).
```

## Model to Predict the Flow

We build an autoregressive transformer model to predict the flow of this dynamical system.

### Tokenization

Fix a radius $`R > 0`$ and a number of bins $`2^B`$.

Each state coordinate $`x_i`$ and output coordinate $`y_i`$ is quantized over the fixed interval $`[-R,R]`$ by splitting it into $`2^B`$ equally spaced bins, indexed by

```math
0,1,\ldots,2^B-1.
```

Each coordinate is mapped to the index of the bin containing it. To dequantize a token, we map the bin index back to the midpoint of the corresponding bin.

Time is quantized separately over the interval $`[0,2\pi]`$.

An input $`(t,x)`$, where $`x \in \mathbb{R}^6`$, is quantized into the seven-token sequence

```math
U
=
\left(
Q(t),
Q(x_1),
\ldots,
Q(x_6)
\right).
```

The exact flow output is

```math
Y
=
\Psi(t,x)
=
(y_1,\ldots,y_6),
```

and the transformer is trained to predict its quantized representation

```math
\left(
Q(y_1),
\ldots,
Q(y_6)
\right).
```

### Autoregressive Model

Let $`P_\theta`$ denote the conditional distribution represented by the transformer.

The output tokens are generated autoregressively. The first coordinate is generated according to

```math
Q(y_1)
\sim
P_\theta(\,\cdot \mid U).
```

The second coordinate is generated according to

```math
Q(y_2)
\sim
P_\theta(\,\cdot \mid U,Q(y_1)),
```

and in general

```math
Q(y_i)
\sim
P_\theta
\left(
\,\cdot
\mid
U,
Q(y_1),
\ldots,
Q(y_{i-1})
\right).
```

Therefore the joint conditional distribution factors as

```math
P_\theta(Y \mid U)
=
\prod_{i=1}^{6}
P_\theta
\left(
Q(y_i)
\mid
U,
Q(y_1),
\ldots,
Q(y_{i-1})
\right).
```

### Training Objective

The model is trained using teacher forcing and cross-entropy loss.

For a batch of $`N`$ training examples $`(U^{(b)},Y^{(b)})`$, the loss is

```math
\mathcal{L}(\theta)
=
-\frac{1}{6N}
\sum_{b=1}^{N}
\sum_{i=1}^{6}
\log
P_\theta
\left(
Y_i^{(b)}
\mid
U^{(b)},
Y_1^{(b)},
\ldots,
Y_{i-1}^{(b)}
\right).
```

The transformer architecture is controlled by three main parameters:

* `config.width`: token embedding dimension;
* `config.layers`: number of transformer blocks;
* `config.heads`: number of attention heads in each transformer block.

## Evaluating Generalization on Out-of-Distribution Data

The main goal of this project is to study how well the transformer generalizes outside the state distribution used during training.

We fix a tokenization radius $`R`$. This means that the meaning of each state token remains unchanged between training and testing.

During training, initial states are sampled uniformly by volume from the six-dimensional ball

```math
B_{R/2}
=
\left\{
x \in \mathbb{R}^6
:
\|x\|_2 \leq \frac{R}{2}
\right\}.
```

For testing, states are sampled uniformly from the larger ball

```math
B_R
=
\left\{
x \in \mathbb{R}^6
:
\|x\|_2 \leq R
\right\}.
```

In six dimensions, the ratio of these volumes is

```math
\frac{\operatorname{Vol}(B_{R/2})}
{\operatorname{Vol}(B_R)}
=
\left(\frac{1}{2}\right)^6
=
\frac{1}{64}.
```

Thus only approximately $`1.56\%`$ of the volume of $`B_R`$ lies inside the training region $`B_{R/2}`$. Equivalently, approximately $`98.44\%`$ of samples drawn uniformly from $`B_R`$ lie outside the training region.

After generating six output tokens, each token is dequantized to the midpoint of its corresponding bin.

The prediction is compared with the exact continuous flow using mean squared error:

```math
\operatorname{MSE}
=
\frac{1}{6N}
\sum_{b=1}^{N}
\sum_{i=1}^{6}
\left(
\hat y_i^{(b)}
-
y_i^{(b)}
\right)^2.
```

The current experiment trains transformers of varying embedding width and records the resulting out-of-distribution test MSE.

## Decoding

### Stochastic Decoding

The default generation method samples each output token from the categorical distribution predicted by the transformer:

```math
\hat Y_i
\sim
P_\theta
\left(
\,\cdot
\mid
U,
\hat Y_1,
\ldots,
\hat Y_{i-1}
\right).
```

### Greedy Decoding

Greedy decoding instead chooses the most probable output token at every autoregressive step:

```math
\hat Y_i
=
\operatorname*{argmax}_{k}
P_\theta
\left(
k
\mid
U,
\hat Y_1,
\ldots,
\hat Y_{i-1}
\right).
```

Comparing stochastic and greedy decoding allows us to distinguish sampling variability from the predictive quality of the learned conditional distributions.

## Project Structure

```text
FlowTransformer/
├── data.py
├── model.py
├── training.py
├── main.py
├── test.py
└── README.md
```

* `data.py`: sampling, exact flow evaluation, quantization, and dequantization.
* `model.py`: transformer architecture and autoregressive generation.
* `training.py`: cross-entropy loss and training loop.
* `main.py`: width-scaling and OOD generalization experiments.
* `test.py`: automated tests for the data, model, and training code.

## Requirements

The main dependencies are:

```text
torch
matplotlib
pytest
```

## Running the Code

Run the main experiment with

```bash
python main.py
```

Run the test suite with

```bash
python -m pytest test.py -v
```

The width-scaling experiment saves numerical results to

```text
width_vs_mse.csv
```

and saves the corresponding plot to

```text
width_vs_mse.png
```

## To Do

* Sample test states directly from the shell

```math
\frac{R}{2}
\leq
\|x\|_2
\leq
R
```

to obtain a purely out-of-distribution test set.

* Compare stochastic and greedy decoding.
* Compare the transformer against simpler neural-network baselines.
* Add further OOD evaluation metrics.
* Add a Slurm script for running experiments on a GPU cluster.
