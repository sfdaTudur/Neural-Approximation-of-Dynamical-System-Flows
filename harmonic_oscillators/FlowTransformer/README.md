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

The state of the system is

$$
z = (q,v) \in \mathbb{R}^6.
$$

The exact flow map is

```math
\Psi(t,q,v)
=
\left(
q\cos(t) + v\sin(t),
-q\sin(t) + v\cos(t)
\right).

**Model to predict flow**

We build an autoregressive transformer model to predict the flow of this dynamical system.
To do this we quantize the input and output coordinates $t,x_i,y_i$ in $[-R,R]$ by splitting 
$[-R,R]$ into $2^B$ equally spaced bins, indexed from $0,..,2^B -1$. We take each coordinate
to the index of the bin containing it. To de-quantize we move from the bin index to the midpoint
of the bin.

An input $(t,x)$ is quantized into a sequence of tokens $U = Q(t),Q(x_1),..,Q(x_6)$, and the goal
of the model is to predict $Y = \Psi(t,x) = (y_1,..,y_6)$ in its quantized form $Q(y_1),..,Q(y_6)$.
Let $P_{\theta}$ denote the conditional distribution on the tokens obtained from the model. Then we 
sample independently $$Q(y_1) ~ P_{\theta}(-|U), Q(y_2) ~ P_{\theta}(-|U,Q(y_1)), ...$$
to obtain $\hat Y = Q(y_1),..,Q(y_n)$ where $$P_{\theta}(\hat Y | U ) = \prod_{i=0}^{5}P_{\theta}(Q(y_{i+1}) | U,..,Q(y_i)).$$
Train using cross-entropy; for a batch of $B$ training points $(U_b,Y_b)$ we use
$$\mathcal L = - \frac{1}{6B} \sum_{b=1}^B \sum_{i=0}^5 - \log P_{\theta}(Y_{i+1}^b | U^b,Y_1^b,...,Y_{i}^b).$$
The transformer architecture has token embedding dimension config.width, consists of config.layers many Decoder blocks,
and each decoder block contains config.heads many attention heads. 

**Evaluating generalization on OOD data**
We fix a radius $R>0$ for the tokenization scheme above, then during training, we only sample input states
from a ball of radius $\frac{R}{2}$. Then we generate a test dataset by sampling states from a ball of
radius $R$, and test how well the trained neural network predicts the flow map, by analyzing the mean squared
error of the flow values obtained from de-quantizing. When sampling from a ball of radius 1, only $0.5^6$ of the
volume of B_1 lies inside B_{1/2}, so roughly 98% of test samples are OOD.

**To do**
Sample $1/2 \leq |x| \leq 1$ for OOD.
Add greedy sampling to generate() in the model, and compare the two methods.
Greedy decoding uses 
$$\hat Y_i = \mathrm{argmax}_k P_{\theta}(Y_k | U,\hat Y_1,..,\hat Y_{i-1})$$

Add test.py.

Add slurm script for running on HPC.

**Requirements**

**How to run the code**

