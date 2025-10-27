import argparse
import math
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import build_model

def rot_matrix(theta: float, device: str = "cpu") -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    G = torch.tensor([[c, -s],
                      [s,  c]], dtype=torch.float32, device=device)
    return G

def apply_rot_diag(G: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Diagonal SO(2) action on (q,p).
    G: (2,2) rotation
    z: (..., 4) with z[..., :2]=q, z[..., 2:]=p  (row-vectors)
    returns: (..., 4) with (Rq, Rp)
    """
    z = z.view(-1, 4)
    q = z[:, :2] @ G.T
    p = z[:, 2:] @ G.T
    zp = torch.cat([q, p], dim=1)
    return zp.view(*(([] if z.dim()==2 else z.shape[:-1])), 4)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate empirical diagonal SO(2)-equivariance error of NN flow, test empirical error on flow property, test invariance of level sets."
    )
    parser.add_argument("--weights", type=str, default=r"D:\flow_datasets\checkpoints\Ham_FlowI_weights.pth",
                        help="path to saved model state")
    parser.add_argument("--t", type=float, default=1.0, help="time t ∈ R")
    parser.add_argument("--x", type=float, nargs=4, default=[1.0, 0.9, 0.1, 1.0],
                        help="initial state (q1, q2, p1, p2)")
    parser.add_argument("--N", type=int, default=200, help="number of rotation powers")
    parser.add_argument("--theta", type=float, default=math.pi * math.sqrt(2.0),
                        help="irrational multiple of pi for rotation angle")
    parser.add_argument("--n_steps", type=float, default=10000, help="number of time steps to test semigroup property.")
    parser.add_argument("--step_size", type=float, default = 0.0001, help="step size to test semigroup property")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (must be 5->4 MLP to match dataset)
    ckpt = torch.load(args.weights, map_location=device)
    model = build_model(hidden1=ckpt["hidden1"], hidden2=ckpt["hidden2"], device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Base input
    x = torch.tensor(args.x, dtype=torch.float32, device=device)     # (4,)
    t_val = torch.tensor([args.t], dtype=torch.float32, device=device)  # (1,)

    # f(t, x)
    tx = torch.cat([t_val, x], dim=0).view(1, 5)  # (1,5)
    with torch.no_grad():
        base_fx = model(tx).view(4)               # (4,)

    theta = args.theta
    accum = 0.0

    # collect f(t, g^i x) for plotting (q' and p' separately)
    preds_q = []
    preds_p = []

    with torch.no_grad():
        for i in range(1, args.N + 1):
            ang = i * theta
            G = rot_matrix(ang, device=device)  # (2,2)

            # g^i · x (diagonal action)
            xi = apply_rot_diag(G, x.view(1, 4))  # (1,4)

            # f(t, g^i · x)
            tx_i = torch.cat([t_val, xi.view(-1)], dim=0).view(1, 5)
            f_pre = model(tx_i).view(4)  # (4,)

            # g^i · f(t, x)
            f_post = apply_rot_diag(G, base_fx.view(1, 4)).view(4)

            diff = f_pre - f_post
            accum += float((diff * diff).sum())

            # store for visualization
            preds_q.append(f_pre[:2].detach().cpu().numpy())
            preds_p.append(f_pre[2:].detach().cpu().numpy())

    equiv_err = accum / args.N
    print(f"Diagonal SO(2)-equivariance MSE (avg over i=1..{args.N}): {equiv_err:.8f}")
    
    # --- Semigroup property test:  E(s) = || f(t0+s, x) - f(s, f(t0, x)) ||^2 ---
    # assumes: x: (4,), t_val: (1,), base_fx = model([t_val, x]) -> (4,)
    steps = args.step_size * torch.arange(args.n_steps, dtype=torch.float32, device=device)
    
    semigroup_error = 0.0
    with torch.no_grad():
        for s in steps:  # s is a 0-dim tensor (scalar) on 'device'
            # f(t0 + s, x)
            total_time_flow = torch.cat([t_val + s.unsqueeze(0), x], dim=0).view(1, 5)
            eval_time_t = model(total_time_flow).view(4)
    
            # f(s, f(t0, x))
            semigroup_flow = torch.cat([s.unsqueeze(0), base_fx], dim=0).view(1, 5)
            eval_semigroup = model(semigroup_flow).view(4)
    
            d = eval_time_t - eval_semigroup
            semigroup_error += float((d*d).sum())
    
    final_semi_err = semigroup_error / len(steps)
    print(f"MSE for semigroup property (avg over {len(steps)} steps, Δ={args.step_size}): {final_semi_err:.8f}")
                
        

    # --- 3D PCA of all predicted outputs f(t, g^i · x) ---
    # Stack all 4D outputs (q1', q2', p1', p2')
    preds_all = np.hstack([preds_q, preds_p])  # shape (N, 4)

    # PCA via SVD (no sklearn dependency)
    def pca_3d(X: np.ndarray):
        """
        Center X (N,4), compute top-3 PCs via SVD, return Y (N,3) and explained variance ratios.
        """
        Xc = X - X.mean(axis=0, keepdims=True)     #center data, given as matrix
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  #find SVD Xc = U S V^T, V 4x4 orthogonal, S 4x4 diagonal with singular vals
        # Project to first 3 PCs
        Y = Xc @ Vt[:3].T  # (N,3)
        # covariance matrix is C = 1/(N-1) Xc^T Xc, and Xc^T Xc = V S^2 V^T, cols of V are eigenvectors of C.
        # rows of V^T are principal component directions. Take the first 3 principal directions.
        # the multiplication Xc @ Vt[:3].T projects each data point onto first 3 PC
        # Explained variance ratio
        var = (S**2) / (len(X) - 1)   #each squared singular value equals variance along corr principal direction
        evr = var / var.sum()
        return Y, evr[:3]

    Y3, evr3 = pca_3d(preds_all)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed by mpl for 3D)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Y3[:, 0], Y3[:, 1], Y3[:, 2], s=12)

    ax.set_xlabel(f"PC1 ({evr3[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr3[1]*100:.1f}% var)")
    ax.set_zlabel(f"PC3 ({evr3[2]*100:.1f}% var)")
    ax.set_title(f"3D PCA of f(t, g^i · x),  t={args.t},  N={args.N}")
    # Improve aspect look a bit
    max_range = (Y3.max(axis=0) - Y3.min(axis=0)).max()
    centers = (Y3.max(axis=0) + Y3.min(axis=0)) / 2
    ax.set_xlim(centers[0]-max_range/2, centers[0]+max_range/2)
    ax.set_ylim(centers[1]-max_range/2, centers[1]+max_range/2)
    ax.set_zlim(centers[2]-max_range/2, centers[2]+max_range/2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

