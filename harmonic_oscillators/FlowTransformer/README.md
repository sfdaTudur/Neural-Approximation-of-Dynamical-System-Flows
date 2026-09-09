# FlowTransformer for the Harmonic Oscillator

Consider the three-dimensional harmonic oscillator

$$
q''(t) = -q(t),
\qquad q(t) \in \mathbb{R}^3.
$$

Writing

$$
v = q',
$$

the corresponding first-order system is

$$
q' = v,
\qquad
v' = -q.
$$

The state of the system is therefore

$$
z = (q,v) \in \mathbb{R}^6.
$$

The exact flow map is

$$
\Psi(t,q,v)
=
\left(
q\cos(t) + v\sin(t),
-q\sin(t) + v\cos(t)
\right).
$$

## Model to Predict the Flow

We build an autoregressive transformer model to approximate the flow of this dynamical system.

### Tokenization

Fix a radius \(R > 0\) and a number of bins \(2^B\).

Each state coordinate \(x_i\) and output coordinate \(y_i\) is quantized over the fixed interval

$$
[-R,R]
$$

by dividing the interval into \(2^B\) equally spaced bins indexed by

$$
0,1,\ldots,2^B-1.
$$

Each coordinate is mapped to the index of the bin containing it. To dequantize a token, we map the bin index back to the midpoint of the corresponding bin.

Time is quantized separately over

$$
[0,2\pi].
$$

An input \((t,x)\), where \(x \in \mathbb{R}^6\), is therefore represented by the seven-token sequence

$$
U
=
\bigl(
Q(t),
Q(x_1),
\ldots,
Q(x_6)
\bigr).
$$

The goal is to predict the quantized flow output

$$
Y
=
\bigl(
Q(y_1),
\ldots,
Q(y_6)
\bigr),
$$

where

$$
(y_1,\ldots,y_6)
=
\Psi(t,x).
$$

### Autoregressive Factorization

Let \(P_\theta\) denote the conditional distribution represented by the transformer.

The output coordinates are generated autoregressively:

$$
Q(y_1)
\sim
P_\theta(\,\cdot\mid U),
$$

$$
Q(y_2)
\sim
P_\theta(\,\cdot\mid U,Q(y_1)),
$$

and more generally

$$
Q(y_i)
\sim
P_\theta
\left(
\,\cdot\mid
U,Q(y_1),\ldots,Q(y_{i-1})
\right).
$$

Thus the joint conditional distribution factors as

$$
P_\theta(Y\mid U)
=
\prod_{i=1}^{6}
P_\theta
\left(
Q(y_i)
\mid
U,Q(y_1),\ldots,Q(y_{i-1})
\right).
$$

### Training Objective

The model is trained using teacher forcing and cross-entropy loss.

For a batch of \(N\) training examples

$$
\{(U^{(b)},Y^{(b)})\}_{b=1}^{N},
$$

the loss is

$$
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
$$

The transformer architecture is controlled by three main parameters:

* `config.width`: token embedding dimension,
* `config.layers`: number of transformer blocks,
* `config.heads`: number of attention heads in each block.

## Evaluating Generalization on OOD Data

The main goal of the project is to study how well the transformer generalizes outside the state distribution used during training.

We fix a tokenization radius \(R\), so the meaning of each token remains unchanged between training and testing.

During training, initial states are sampled uniformly by volume from the six-dimensional ball

$$
B_{R/2}
=
\left\{
x\in\mathbb{R}^6 :
\|x\|_2\leq \frac{R}{2}
\right\}.
$$

For testing, states are sampled uniformly from

$$
B_R
=
\left\{
x\in\mathbb{R}^6 :
\|x\|_2\leq R
\right\}.
$$

In six dimensions,

$$
\frac{\operatorname{Vol}(B_{R/2})}
{\operatorname{Vol}(B_R)}
=
\left(\frac{1}{2}\right)^6
=
\frac{1}{64}.
$$

Therefore only about \(1.56\%\) of the volume of \(B_R\) lies inside the training region, so approximately \(98.44\%\) of uniformly sampled test states lie outside \(B_{R/2}\).

The generated output tokens are dequantized to bin midpoints and compared with the exact continuous flow values using mean squared error:

$$
\operatorname{MSE}
=
\frac{1}{6N}
\sum_{b=1}^{N}
\sum_{i=1}^{6}
\left(
\hat y_i^{(b)}-y_i^{(b)}
\right)^2.
$$

The current experiments compare this test MSE across transformers of different embedding widths.

## Decoding

The model supports stochastic autoregressive decoding by sampling from each conditional categorical distribution.

A second evaluation method is greedy decoding, where at each step we choose the most likely output bin:

$$
\hat Y_i
=
\operatorname*{argmax}_k
P_\theta
\left(
k
\mid
U,
\hat Y_1,
\ldots,
\hat Y_{i-1}
\right).
$$

Comparing stochastic and greedy decoding allows us to distinguish sampling variability from the predictive quality of the learned conditional distributions.

## To Do

* Sample test states directly from the shell

$$
\frac{R}{2}
\leq
\|x\|_2
\leq
R
$$

to obtain a purely OOD test set.

* Compare stochastic and greedy decoding.
* Add additional OOD evaluation metrics.
* Add automated tests with `pytest`.
* Add a Slurm script for single-GPU HPC experiments.
* Compare the transformer against simpler neural-network baselines.

## Requirements

The project requires Python and PyTorch. Additional packages used for testing and plotting include:

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

The width-scaling experiment writes its numerical results to

```text
width_vs_mse.csv
```

and saves the corresponding plot as

```text
width_vs_mse.png
```
