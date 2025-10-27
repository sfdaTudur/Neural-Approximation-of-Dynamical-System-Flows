import argparse
import math
import os
from typing import Tuple

import torch

# ----- Problem setup (torch) -----

def gradV(q: torch.Tensor, lam: float) -> torch.Tensor:
    """
    q: (..., 2) tensor
    returns ∇_q V(q) = q + 4 λ ||q||^2 q, shape (..., 2)
    """
    r2 = (q * q).sum(dim=-1, keepdim=True)
    return q * (1.0 + 4.0 * lam * r2)

def H(q: torch.Tensor, p: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Hamiltonian for monitoring (optional).
    q, p: (..., 2)
    returns (...,)
    """
    r2 = (q * q).sum(dim=-1)
    return 0.5 * (p * p).sum(dim=-1) + 0.5 * r2 + lam * (r2 ** 2)

# ----- One leapfrog step: (q_n, p_n) -> (q_{n+1}, p_{n+1}) -----

@torch.no_grad()
def leapfrog_step(q: torch.Tensor, p: torch.Tensor, h: float, lam: float):
    """
    Velocity–Verlet step for separable H = T(p) + V(q).
    q, p: (..., 2)  (broadcastable batch)
    h: time step (can be negative or any real)
    """
    # half kick
    p_half = p - 0.5 * h * gradV(q, lam)
    # drift
    q_next = q + h * p_half
    # half kick
    p_next = p_half - 0.5 * h * gradV(q_next, lam)
    return q_next, p_next

# ----- Flow map approximation Φ̂_t via repeated leapfrog -----

@torch.no_grad()
def flow_to_time(q0: torch.Tensor, p0: torch.Tensor, t: float, h: float, lam: float):
    """
    Approximate Φ_t(q0,p0) using leapfrog with base step h and a final remainder step.
    t can be positive or negative.
    """
    if abs(t) < 1e-15:
        return q0, p0
    s = 1.0 if t >= 0.0 else -1.0
    t_abs = abs(t)
    N = int(math.floor(t_abs / abs(h)))
    step = s * abs(h)
    r = s * (t_abs - N * abs(h))  # remainder in [-|h|, |h|]

    q, p = q0, p0
    for _ in range(N):
        q, p = leapfrog_step(q, p, step, lam)
    if abs(r) > 1e-15:
        q, p = leapfrog_step(q, p, r, lam)
    return q, p

# ----- Dataset generation -----

def sample_uniform_ball_R4(num: int, radius: float, g: torch.Generator) -> torch.Tensor:
    """
    Samples num points uniformly from the ball of radius 'radius'. Return sample
    as a (num,4) tensor.
    """
    x = torch.randn(num, 4, generator=g)  #draw num iid standard normal vectors
    norms = x.norm(dim=1, keepdim=True)  #compute each vectors norm
    x = x / norms.clamp_min(1e-12)    #normalize each random vector
    u = torch.rand(num, 1, generator=g)  #iid uniforms on (0,1) for radii. U = Uniform(0,1)
    r = radius * u.pow(1.0 / 4.0)    #R' = U**(1/4) has pdf f(r) = 4r**3, vol element for uniform sampling ~ r**3 dr
    return r * x  # (num, 4)

def make_dataset(
    num_x: int,
    num_t: int,
    t0: float,
    dt: float,
    h_int: float,
    lam: float,
    radius: float,
    seed: int = 123,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build pairs ((t, q1, q2, p1, p2), Φ(t,(q,p))).

    - Sample (q,p) uniformly from 4D ball of radius 'radius'.
    - Times t_k = t0 + k*dt, k = 0..num_t-1.
    - For each t_k, apply leapfrog-based flow approximation from (q,p).

    Returns:
      inputs:  (N, 5) columns = (t, q1, q2, p1, p2)
      targets: (N, 4) columns = (q1', q2', p1', p2')  ≈ Φ(t,(q,p))
      where N = num_x * num_t
    """
    g = torch.Generator().manual_seed(seed)

    # sample initial states in R^4, then split into q,p
    X0 = sample_uniform_ball_R4(num_x, radius, g)  # (num_x, 4)
    q0 = X0[:, 0:2]
    p0 = X0[:, 2:4]

    # time values
    t_vals = t0 + dt * torch.arange(num_t, dtype=torch.float32)  # (num_t,)

    # Repeat states for each time, and compute flows time-by-time for efficiency
    inputs_list = []
    targets_list = []
    for t in t_vals.tolist():
        # apply flow to the whole batch at this t
        qk, pk = flow_to_time(q0, p0, t, h_int, lam)

        # record inputs/targets
        t_col = torch.full((num_x, 1), float(t))
        inp = torch.cat([t_col, q0, p0], dim=1)          # (num_x, 5)
        tar = torch.cat([qk, pk], dim=1)                 # (num_x, 4)

        inputs_list.append(inp)
        targets_list.append(tar)

    inputs = torch.vstack(inputs_list)   # (num_t*num_x, 5)
    targets = torch.vstack(targets_list) # (num_t*num_x, 4)
    return inputs, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_x", type=int, default=5000, help="number of distinct initial (q,p) states")
    parser.add_argument("--num_t", type=int, default=200, help="number of time values")
    parser.add_argument("--t0", type=float, default=0.0, help="first time value")
    parser.add_argument("--dt", type=float, default=0.05, help="spacing between time values")
    parser.add_argument("--h_int", type=float, default=0.01, help="internal leapfrog step size h")
    parser.add_argument("--lam", type=float, default=0.2, help="lambda in V(q) = 1/2||q||^2 + lam||q||^4")
    parser.add_argument("--radius", type=float, default=2.0, help="sampling radius for (q,p) in R^4 (4D ball)")
    parser.add_argument("--out", type=str, default=r"D:\flow_datasets\Ham_FlowI_train_data.pt", help="output path")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    inputs, targets = make_dataset(
        num_x=args.num_x,
        num_t=args.num_t,
        t0=args.t0,
        dt=args.dt,
        h_int=args.h_int,
        lam=args.lam,
        radius=args.radius,
        seed=args.seed,
    )

    torch.save({"inputs": inputs, "targets": targets,
                "meta": {"lam": args.lam, "h_int": args.h_int, "t0": args.t0, "dt": args.dt}}, args.out)

    N = inputs.shape[0]
    print(f"Saved dataset with N={N} samples to {args.out}")

if __name__ == "__main__":
    main()
