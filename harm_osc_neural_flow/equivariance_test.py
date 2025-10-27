import argparse
import math
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import build_model

def rot_matrix(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

def apply_rot(G: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """G: (2,2), x: (..., 2) -> (..., 2)"""
    return x @ G.T

def main():
    parser = argparse.ArgumentParser(description="Evaluate empirical SO(2)-equivariance error of f")
    parser.add_argument("--weights", type=str, default=r"D:\flow_datasets/checkpoints/model_harm_osc.pth", help="path to saved model state")
    parser.add_argument("--t", type=float, default=1, help="time t ∈ R")
    parser.add_argument("--x", type=float, nargs=2, default=[1, 0.9], help="initial point x ∈ R^2 (x1 x2)")
    parser.add_argument("--N", type=int, default=200, help="number of rotation powers")
    parser.add_argument("--theta", type=float, default=math.pi * math.sqrt(2.0), help="irrational multiple of pi for rotation angle")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.weights, map_location=device)
    model = build_model(hidden1=ckpt["hidden1"], hidden2=ckpt["hidden2"], device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = torch.tensor(args.x, dtype=torch.float32)  # (2,)
    t_val = torch.tensor([args.t], dtype=torch.float32)  # (1,)

    # f(t, x)
    tx = torch.cat([t_val, x], dim=0).view(1, 3).to(device)
    base_fx = model(tx).detach().cpu().view(2)

    theta = args.theta
    accum = 0.0
    
    #collect the f(t, g^ix) for plotting
    preds = []
    
    for i in range(1, args.N + 1):
        ang = i * theta
        G = rot_matrix(ang)  # (2,2)

        # g^i · x
        xi = apply_rot(G, x.view(1, 2))  # (1,2)

        # f(t, g^i · x)
        tx_i = torch.cat([t_val, xi.view(-1)], dim=0).view(1, 3).to(device)
        f_pre = model(tx_i).detach().cpu().view(2)
        
        preds.append(f_pre.numpy())   #store prediction f(t, g^i x )

        # g^i · f(t, x)
        f_post = apply_rot(G, base_fx.view(1, 2)).view(2)

        diff = f_pre - f_post
        accum += float(torch.dot(diff, diff))

    equiv_err = accum / args.N
    print(f"SO(2)-equivariance MSE (average over i=1..{args.N}): {equiv_err:.8f}")
    
    # ---plot of all points f(t, g^i x) ---
    preds = np.vstack(preds)  # shape (N, 2)
    plt.figure(figsize=(6, 6))
    plt.scatter(preds[:, 0], preds[:, 1], s=10, marker='o')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title(f"Points f(t, g^i x) for t={args.t}, x={tuple(args.x)}")
    plt.show()

if __name__ == "__main__":
    main()
