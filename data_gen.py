import argparse
import math
import os
from typing import Tuple

import torch

def psi_flow(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Ψ(t, (a,b)) = (a cos t - b sin t, a sin t + b cos t)
    flow of the harmonic oscillator z' = Jz with J the matrix
    representing multiplication by i as an R-linear map.

    t: shape (N,) or (N,1)
    x: shape (N,2)
    returns: shape (N,2)
    """
    t = t.view(-1, 1)
    a = x[:, 0:1]
    b = x[:, 1:2]
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    y1 = a * cos_t - b * sin_t
    y2 = a * sin_t + b * cos_t
    return torch.cat([y1, y2], dim=1)

def make_dataset(
    num_x: int, num_t: int, t0: float, dt: float, radius: float, seed: int = 123
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    inputs: shape (N, 3) with columns (t, x1, x2)
    targets: shape (N, 2) with Ψ(t, x)

    x is sampled uniformly from the disk of radius `radius` in R^2.
    t is a sequence t_k = t0 + k*dt for k = 0..num_t-1, paired with all x.
    """
    g = torch.Generator().manual_seed(seed)   #random number generator

    # sample x in a disk
    angles = 2 * math.pi * torch.rand(num_x, generator=g)
    radii = radius * torch.sqrt(torch.rand(num_x, generator=g))
    x = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)  #shape (num_x,2)

    t_vals = t0 + dt * torch.arange(num_t, dtype=torch.float32)  #shape (num_t,)

    # cartesian product of t and x
    T = t_vals.repeat_interleave(num_x)  # repeat each time value num_x times, shape(num_t*num_x,)
    X = x.repeat(num_t, 1)               #stack all x points num_t times, shape (num_t*num_x, 2)

    inputs = torch.cat([T.view(-1, 1), X], dim=1)  # (N,3)
    targets = psi_flow(T, X)                        # (N,2)
    return inputs, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_x", type=int, default=5000, help="number of distinct x points")
    parser.add_argument("--num_t", type=int, default=200, help="number of time steps")
    parser.add_argument("--t0", type=float, default=0.0, help="initial time")
    parser.add_argument("--dt", type=float, default=0.05, help="time step size")
    parser.add_argument("--radius", type=float, default=2.0, help="sampling radius for x in R^2")
    parser.add_argument("--out", type=str, default=r"D:\flow_datasets\harmonic_oscillator_train_data.pt", help="output path for dataset")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    inputs, targets = make_dataset(args.num_x, args.num_t, args.t0, args.dt, args.radius, args.seed)

    torch.save({"inputs": inputs, "targets": targets}, args.out)

    N = inputs.shape[0]
    print(f"Saved dataset with N={N} samples to {args.out}")

if __name__ == "__main__":
    main()