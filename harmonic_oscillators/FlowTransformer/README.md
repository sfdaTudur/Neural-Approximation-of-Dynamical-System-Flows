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

**Model to predict flow**

We build an autoregressive transformer model to predict the flow of this dynamical system, and study how well the transformer generalizes outside the state distribution used during training. To do this we quantize the input and output coordinates as follows: 

Fix a radius $R > 0$ and a number of bins $2^B$. Each state coordinate $x_i$ and output coordinate $y_i$ is quantized over the fixed interval $[-R,R]$ by splitting the interval into $2^B$ equally spaced bins indexed by

```math
0,1,\ldots,2^B-1.
```

Each coordinate is mapped to the index of the bin containing it. To dequantize we map the bin index to the midpoint of the corresponding bin. Time is quantized separately over the interval $[0,2\pi]$.

An input $(t,x)$, where $x \in \mathbb{R}^6$, is represented by the seven-token sequence

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

The exact continuous flow output is

```math
Y
=
\Psi(t,x)
=
(y_1,\ldots,y_6).
```

The transformer is trained to predict its quantized representation

```math
\left(
Q(y_1),
\ldots,
Q(y_6)
\right).
```

**Autoregressive model**

Let $P_\theta$ denote the conditional distribution represented by the transformer. In the stochastic method of generation, the output coordinates are generated according to
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
The joint conditional distribution factors as

```math
P_\theta(Y\mid U)
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
We test the stochastic method against greedy generation, which chooses the most probable output token at each autoregressive step

```math
 Q(y_i)
=
\underset{k \in \{0,..,2^B-1}}{\mathrm{argmax}}
\;
P_\theta
\left(
Q(y_i)=k
\mid
U,
Q(y_1),
\ldots,
Q(y_{i-1})
\right).
```


**Training Objective**

The model is trained using cross-entropy loss.

For a batch of $N$ quantized training examples $(U^{(b)},Y^{(b)})$, the loss is

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

**Evaluating Generalization on Out-of-Distribution Data**

The main goal of this project is to study how well the transformer generalizes outside the state distribution used during training. We fix a tokenization radius $R$ so that the meaning of each token remains unchanged between training and testing. During training, initial states are sampled uniformly by volume from the six-dimensional ball

```math
B_{R/2}
=
\left\{
x \in \mathbb{R}^6
:
\|x\|_2
\leq
\frac{R}{2}
\right\}.
```

For testing, states are sampled uniformly from the larger ball

```math
B_R
=
\left\{
x \in \mathbb{R}^6
:
\|x\|_2
\leq
R
\right\}.
```

In six dimensions, the ratio of these volumes is

```math
\frac{\mathrm{Vol}(B_{R/2})}
{\mathrm{Vol}(B_R)}
=
\left(
\frac{1}{2}
\right)^6
=
\frac{1}{64}.
```
Thus only approximately $1.56%$ of the volume of $B_R$ lies inside the training region $B_{R/2}$; approximately $98.44%$ of states sampled uniformly from $B_R$ lie outside the training region. 

After generating six output tokens, each predicted token is dequantized to the midpoint of its corresponding bin. The prediction is compared with the exact continuous flow using mean squared error:

```math
\mathrm{MSE}
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

The current experiment trains transformers of varying embedding width while keeping the number of layers and attention heads fixed. For each width:

1. A new transformer is initialized.
2. The transformer is trained on the same fixed training dataset sampled from $B_{R/2}$.
3. The trained model is evaluated on the same fixed test dataset sampled from $B_R$. We evaluate using both stochastic and greedy generation of output tokens.
4. The test MSE is recorded for both stochastic and greedy generation.
5. The model is deleted before training the next width.

The experiment writes

```text
width,Stochastic mse,Greedy mse
```
to
```text
width_vs_mse.csv
```
and saves the corresponding plot as
```text
width_vs_mse.png
```

**Project Structure**

```text
FlowTransformer/
├── data.py
├── model.py
├── training.py
├── main.py
├── test.py
├── slurm_script.sh
└── README.md
```

* data.py: sampling, exact flow evaluation, quantization, and dequantization.
* model.py: transformer architecture and autoregressive generation.
* training.py: cross-entropy loss and training loop.
* main.py: width-scaling and OOD generalization experiments.
* test.py: automated tests for the data, model, and training code.
* slurm_script.sh: slurm script used to submit the experiment as a batch job on Isambard HPC.

**Requirements**

The main dependencies are

```text
torch
matplotlib
pytest
```

**Running the Code Locally**

Run the main experiment with

```bash
python main.py
```

Run the test suite with

```bash
python -m pytest test.py -v
```

**Running the Code on Isambard-AI with Slurm**

Initialize miniforge
```bash
source ~/miniforge3/bin/activate
```
get an interactive GPU node
```bash 
srun --nodes=1 --gpus=1 --time=00:30:00 --pty /bin/bash --login
```
and once the compute node starts, run
```bash
source ~/miniforge3/bin/activate
conda create --name harmosc-hpc --channel conda-forge python=3.12 pytorch matplotlib pytest
conda activate harmosc-hpc
```
verfy that the required packages can be imported
```bash
python -c "import torch, matplotlib, pytest; print('Packages installed successfully')"
```
check whether PyTorch can access CUDA GPU
```bash
python -c "import torch; print('CUDA:available', torch.cuda.is_available())"
```
Submit the script on HPC from the project directory:
```bash
conda activate harmosc-hpc
sbatch slurm_script.sh
```
