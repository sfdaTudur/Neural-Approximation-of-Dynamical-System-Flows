import torch
import math
import csv
import os
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("data", exist_ok=True)
os.makedirs("data/plots", exist_ok=True)

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


#Use GPU if device has one, else use CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


#ReLU neural net with specified depth, used to approximate the flow of a dynamical system
class NeuralNetwork(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.depth = depth
        self.width = width

        self.first_layer = nn.Sequential(
            nn.Linear(7, width),
            nn.ReLU()
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU()
            )
            for _ in range(depth)
        ])

        self.final_layer = nn.Linear(width, 6)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.final_layer(x)


#---------------------Training data for the flow of the Harmonic oscillator-------------

def psi_flow_3d_oscillator(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Flow for q' = v, v' = -q in R^3.

    t: shape (N,) or (N,1)
    z: shape (N,6), with z[:,0:3] = q and z[:,3:6] = v

    returns: shape (N,6)
    """
    t = t.view(-1, 1)
    q = z[:, 0:3]
    v = z[:, 3:6]

    cos_t = torch.cos(t)
    sin_t = torch.sin(t)

    q_new = q * cos_t + v * sin_t
    v_new = -q * sin_t + v * cos_t

    return torch.cat([q_new, v_new], dim=1)



def sample_ball_6d(num_samples, R):
    z_dir = torch.randn(num_samples, 6, device=device, dtype=torch.float32)
    z_dir = z_dir / torch.norm(z_dir, dim=1, keepdim=True)

    u = torch.rand(num_samples, 1, device=device, dtype=torch.float32)
    r = R * u ** (1.0 / 6.0)

    return r * z_dir



def make_training_data(batch_size, R, noise_std=0.01):
    #t ~ Uniform[0,2\pi]
    T = 2 * math.pi *torch.rand(batch_size,device = device, dtype=torch.float32)

    # Sample z = (q, v) uniformly from the 6D ball B(0, R) in R^6.
    # For uniform-in-volume sampling in R^6, use radius r = R * u^(1/6).
    z = sample_ball_6d(batch_size, R)

    # Input is (t, q, v) in R^7
    inputs = torch.cat((T.view(-1, 1), z), dim=1)

    clean_targets = psi_flow_3d_oscillator(T, z)

    # Add small iid Gaussian noise to the target
    noise = noise_std * torch.randn_like(clean_targets)
    noisy_targets = clean_targets + noise

    return inputs, noisy_targets






#--------------------------------Training--------------------------------------------
def train_model(model, depth, width, criterion, optimizer, dl, radius, batch_size, no_of_epochs):
    model.train()
    for epoch in range(1, no_of_epochs + 1):
        running_loss = 0.0
        for batch_inputs, batch_targets in dl:
            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)
        epoch_loss = running_loss / len(dl.dataset)
        print(f"Epoch {epoch:03d} | MSE: {epoch_loss:.6f}")

    # Save weights
    save_path = f"data/model_weights_depth_{depth}_width_{width}.pth"

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "depth": depth,
            "width": width,
            "radius": radius,
            "batch_size": batch_size,
            "epochs": no_of_epochs,
        },
        save_path,
    )

    print(f"Saved model weights to {save_path}")


    

#---------------------------------Set parameters for training------------------------------
max_depth = 15   
max_width = 32   
dataset_size = 10000   
radius = 5    #radius of sampled (x,y)
batch_size = 2048   #size of one batch for gradient descent
no_of_epochs = 50   #epochs for training
inputs, targets = make_training_data(dataset_size, radius,noise_std=0.01)
ds = torch.utils.data.TensorDataset(inputs, targets)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)



for depth in range(1,max_depth):
    for width in range(1, max_width):
        model = NeuralNetwork(depth, width).to(device)
        criterion = nn.MSELoss()
        #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)   # Stochastic Gradient Descent
        train_model(model, depth, width, criterion, optimizer, dl, radius, batch_size, no_of_epochs)
    




#-----------------------------Equivariance test--------------------------------


def random_so3(batch_size, device=device):
    """
    Returns batch_size random SO(3) matrices.
    """
    A = torch.randn(batch_size, 3, 3, device=device)
    Q, R = torch.linalg.qr(A)

    # Fix signs to make Q Haar-ish
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    signs = torch.sign(diag)
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(-2)

    # Ensure determinant +1
    det = torch.linalg.det(Q)
    Q[det < 0, :, 0] *= -1

    return Q


def apply_so3_to_state(G, z):
    """
    G: shape (N,3,3)
    z: shape (N,6)
    returns: shape (N,6)
    """
    q = z[:, 0:3].unsqueeze(-1)
    v = z[:, 3:6].unsqueeze(-1)

    Gq = torch.bmm(G, q).squeeze(-1)
    Gv = torch.bmm(G, v).squeeze(-1)

    return torch.cat([Gq, Gv], dim=1)



def empirical_equivariance_so3(model, num_samples, R):
    model.eval()

    with torch.no_grad():
        T = 2 * math.pi * torch.rand(num_samples, 1, device=device)

        #sample uniformly from B(0,R) in R^6
        z = sample_ball_6d(num_samples, R)

        G = random_so3(num_samples, device=device)

        Gz = apply_so3_to_state(G, z)

        inp_Gz = torch.cat([T, Gz], dim=1)
        inp_z = torch.cat([T, z], dim=1)

        f_Gz = model(inp_Gz)
        f_z = model(inp_z)

        G_f_z = apply_so3_to_state(G, f_z)

        error = ((f_Gz - G_f_z) ** 2).mean()

    return error.item()


#-----------------Using test data---------------------------------

def test_flow_error(model, num_samples, R):
    model.eval()

    with torch.no_grad():
        T = 2 * math.pi * torch.rand(num_samples, 1, device=device)
        z = sample_ball_6d(num_samples, R)

        inputs = torch.cat((T, z), dim=1)
        clean_targets = psi_flow_3d_oscillator(T, z)

        preds = model(inputs)

        error = ((preds - clean_targets) ** 2).mean()

    return error.item()






#Store (depth, width,equivariance)- and (depth,width,test_error)-tables to study double-descent
num_samples = 10000
test_radii = [5,10,20,40]


with open("data/empirical_equivariance_vs_depth_width.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["depth", "width", "R_test", "eq_error", "test_error"]) 

    for depth in range(1, max_depth):
        for width in range(1, max_width):
            model = NeuralNetwork(depth, width).to(device)

            checkpoint = torch.load(
                f"data/model_weights_depth_{depth}_width_{width}.pth",
                map_location=device
            )
            model.load_state_dict(checkpoint["state_dict"])


            for R_test in test_radii:
                eq_error = empirical_equivariance_so3(model, num_samples, R_test)
                test_error = test_flow_error(model, num_samples, R_test)
                writer.writerow([depth, width, R_test, eq_error, test_error])



#------------------plot results---------------------------------------

def plot_error_surface(
    dataframe,
    R_fixed,
    error_column,
    output_directory="data/plots",
    use_log10=True,
):
    """
    Plot error as a function of depth and width for one fixed test radius.

    error_column must be either:
        "eq_error"
        "test_error"
    """
    if error_column not in {"eq_error", "test_error"}:
        raise ValueError(
            "error_column must be either 'eq_error' or 'test_error'"
        )

    # Select rows corresponding to the chosen test radius
    selected = dataframe[dataframe["R_test"] == R_fixed].copy()

    if selected.empty:
        available_radii = sorted(dataframe["R_test"].unique())
        raise ValueError(
            f"No data found for R_test={R_fixed}. "
            f"Available radii are {available_radii}."
        )

    # Construct a depth-by-width grid
    error_grid = selected.pivot(
        index="depth",
        columns="width",
        values=error_column,
    )

    error_grid = error_grid.sort_index()
    error_grid = error_grid.sort_index(axis=1)

    depths = error_grid.index.to_numpy()
    widths = error_grid.columns.to_numpy()

    width_mesh, depth_mesh = np.meshgrid(widths, depths)
    error_values = error_grid.to_numpy(dtype=float)

    # Logarithms usually make error surfaces much easier to see,
    # particularly when the error varies over several orders of magnitude.
    if use_log10:
        positive_floor = np.finfo(float).tiny
        plotted_values = np.log10(
            np.maximum(error_values, positive_floor)
        )
        z_label = f"log10({error_column})"
    else:
        plotted_values = error_values
        z_label = error_column

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        depth_mesh,
        width_mesh,
        plotted_values,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
    )

    ax.set_xlabel("Depth")
    ax.set_ylabel("Width")
    ax.set_zlabel(z_label)
    ax.set_title(
        f"{error_column} versus depth and width, R_test={R_fixed}"
    )

    fig.colorbar(
        surface,
        ax=ax,
        shrink=0.7,
        pad=0.1,
        label=z_label,
    )

    fig.tight_layout()

    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(
        output_directory,
        f"{error_column}_surface_R_{R_fixed}.png",
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved plot to {output_path}")


# Read the generated CSV file
results = pd.read_csv(
    "data/empirical_equivariance_vs_depth_width.csv"
)

# Plot both quantities for fixed radii
for R_fixed in sorted(results["R_test"].unique()):
    plot_error_surface(
        results,
        R_fixed=R_fixed,
        error_column="eq_error",
    )

    plot_error_surface(
        results,
        R_fixed=R_fixed,
        error_column="test_error",
    )