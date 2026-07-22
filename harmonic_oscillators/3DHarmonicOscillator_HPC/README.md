## Overview

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
```

This flow is equivariant under the diagonal action of $SO(3)$.

For a rotation $R \in SO(3)$, the action on the state is

```math
R \cdot (q,v) = (Rq,Rv),
```

and the flow satisfies

```math
\Psi(t,Rq,Rv)
=
R \cdot \Psi(t,q,v).
```


## Experiment

Fully connected ReLU neural networks of varying depths and widths are trained to approximate the exact flow map.

Each network takes seven inputs:

* one time coordinate (t);
* three position coordinates (q);
* three velocity coordinates (v).

The network produces six outputs representing the predicted position and velocity after time (t).

The experiment measures:

1. the error between the neural-network prediction and the exact flow;
2. the empirical (SO(3))-equivariance error;
3. the change in these errors for out-of-distribution states sampled from balls of increasing radius.

For each neural-network architecture, the program:

1. trains the model using a fixed training dataset;
2. calculates the flow and equivariance errors;
3. appends the results to a CSV file;
4. deletes the trained model and clears unused memory;
5. moves to the next architecture.

The same training dataset is used for every architecture so that the comparisons between architectures are consistent.



## Repository contents

```text
3DHarmonicOscillator_HPC/
├── HarmOsc_HPC.py
├── README.md
└── slurm_script.sh
```

### `HarmOsc_HPC.py`

Contains the neural-network model, training procedure, exact harmonic-oscillator flow, (SO(3))-equivariance test, out-of-distribution error calculation, CSV output, and interactive plotting code.

### `slurm_script.sh`

A Slurm batch-submission script for running `HarmOsc_HPC.py` on the Isambard HPC system.

The script specifies the requested HPC resources, activates the Python environment, and launches the experiment on a compute node. Review the `#SBATCH` resource settings, such as the time limit and number of GPUs, before submitting the job.

## Requirements

The program requires Python 3.10 or later and the following third-party packages:

| Package   | Purpose                                                          |
| --------- | ---------------------------------------------------------------- |
| `pytorch` | Neural-network training, tensor operations, and GPU acceleration |
| `numpy`   | Numerical operations and preparation of plotting data            |
| `pandas`  | Reading and processing the results CSV file                      |
| `plotly`  | Generating interactive three-dimensional HTML plots              |

The modules `math`, `csv`, `os`, and `gc` are part of the Python standard library and do not need to be installed separately.

## Local installation with Conda

Create a new Conda environment:

```bash
conda create --name harmosc-hpc --channel conda-forge python=3.12 numpy pandas plotly pytorch
```

Activate the environment:

```bash
conda activate harmosc-hpc
```

Verify that the required packages can be imported:

```bash
python -c "import torch, numpy, pandas, plotly; print('Packages installed successfully')"
```

Check whether PyTorch can access a CUDA GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The program automatically uses a CUDA GPU when one is available. Otherwise, it runs on the CPU.

## Running locally

From the `3DHarmonicOscillator_HPC` directory, run:

```bash
python HarmOsc_HPC.py
```

Because the program trains 224 different neural-network architectures, completing the full experiment may take a substantial amount of time on a CPU.



## Running on Isambard with Slurm

The file `slurm_script.sh` is provided to submit the experiment as a batch job through the Slurm workload manager.

Activate the environment and submit the script from the project directory:

```bash
conda activate harmosc-hpc
sbatch slurm_script.sh
```

Slurm will place the job in the queue and run it when the requested compute resources become available.



## Output

The program creates the following directories automatically:

```text
data/
└── plots/
```

Numerical results are saved to:

```text
data/empirical_equivariance_vs_depth_width.csv
```

The CSV file contains:

* network depth;
* network width;
* test radius;
* empirical equivariance error;
* flow prediction error.

Interactive Plotly surfaces are generated for each test radius and saved in:

```text
data/plots/
```

The generated `.html` files can be opened in a web browser. Each plot shows the relationship between network depth, network width, and either the flow error or the empirical equivariance error.
