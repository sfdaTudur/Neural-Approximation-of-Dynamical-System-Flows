# Neural Approximation of Dynamical-System Flows

This repository contains experiments on learning the flows of dynamical systems using neural networks. 

The system currently studied is the three-dimensional harmonic oscillator. The harmonic oscillator 
provides a simple dynamical system with a known exact solution and a natural rotational symmetry, making  
it a useful setting for studying how neural-network architecture and model size affect approximation
accuracy, equivariance, and generalization outside the training distribution.

The repository currently contains two experiments.

## Fully connected Neural Networks

```bash
3DHarmonicOscillator_HPC/
```
Fully connected ReLU networks of varying depth and width are trained to approximate the exact flow map
of the harmonic oscillator. The experiment measures both flow-prediction error and empirical SO(3)-equivariance
error. 

Models are trained on states sampled from a fixed-radius ball and evaluated at multiple test radii, allowing
both in-distribution and out-of-distribution behaviour to be studied. Errors are also compared with model parameter count to investigate how approximation quality scales with network size. 

The experiment is designed to run either locally or as a Slurm batch job on an HPC system. See
```bash
3DHarmonicOscillator_HPC/README.md
```
for the mathematical setup, experiment details, and instructions for reproducing results.

## Autoregressive transformer

```bash
FlowTransformer/
```
The second experiment represents the harmonic-oscillator flow as a sequence-modelling problem.

Continuous coordinates are quantized into discrete tokens, and an autoregressive transformer is trained to predict the six coordinates of the evolved state. The model is evaluated using both stochastic sampling and greedy generation.

The experiment studies how prediction error and out-of-distribution generalization change as the transformer architecture is varied. See 
```
FlotTransformer/README.md
```
for details of the tokenization scheme, transformer architecture, training objective, and evaluation procedure.

### Questions explored

* How does neural-network size affect approximation error?
* Does increasing the model capacity improve empirical rotational equivariance? In other words,
  do higher capacity models learn the underlying symmetries of the dynamical system?
* How do models trained on a bounded state distribution behave outside that distribution?
* How do standard fully connected networks compare with a discrete autoregressive formulation
  of the same flow-learning problem?

### Results

Numerical results and generated plots are stored within the corresponding experiment directories.

The fully connected network experiment produces interactive Plotly visualizations of flow and 
equivariance error as functions of network depth, width, and parameter count.
