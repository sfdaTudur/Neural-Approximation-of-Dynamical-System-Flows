import torch
import math
import csv
import os
import gc
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
os.makedirs("data/plots", exist_ok=True)

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# ReLU neural net used to approximate the flow of a dynamical system.
class NeuralNetwork(nn.Module):
    def __init__(self, depth, width):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(7, width),
            nn.ReLU(),
        )

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
            )
            for _ in range(depth)
        ])

        self.final_layer = nn.Linear(width, 6)

    def forward(self, x):
        x = self.first_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        return self.final_layer(x)


# ---------------- Harmonic-oscillator flow and data ----------------

def psi_flow_3d_oscillator(t, z):
    t = t.view(-1, 1)

    q = z[:, 0:3]
    v = z[:, 3:6]

    cos_t = torch.cos(t)
    sin_t = torch.sin(t)

    q_new = q * cos_t + v * sin_t
    v_new = -q * sin_t + v * cos_t

    return torch.cat([q_new, v_new], dim=1)


def sample_ball_6d(num_samples, radius):
    z_direction = torch.randn(
        num_samples,
        6,
        device=device,
        dtype=torch.float32,
    )

    z_direction = z_direction / torch.norm(
        z_direction,
        dim=1,
        keepdim=True,
    )

    u = torch.rand(
        num_samples,
        1,
        device=device,
        dtype=torch.float32,
    )

    sampled_radius = radius * u ** (1.0 / 6.0)

    return sampled_radius * z_direction


def make_training_data(num_samples, radius, noise_std=0.01):
    times = 2 * math.pi * torch.rand(
        num_samples,
        device=device,
        dtype=torch.float32,
    )

    states = sample_ball_6d(num_samples, radius)

    inputs = torch.cat(
        (times.view(-1, 1), states),
        dim=1,
    )

    clean_targets = psi_flow_3d_oscillator(times, states)
    noisy_targets = clean_targets + noise_std * torch.randn_like(clean_targets)

    return inputs, noisy_targets


# ---------------- Training ----------------

def train_model(model, criterion, optimizer, data_loader, number_of_epochs):
    model.train()

    for epoch in range(1, number_of_epochs + 1):
        running_loss = 0.0

        for batch_inputs, batch_targets in data_loader:
            optimizer.zero_grad()

            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)

        print(
            f"Epoch {epoch:03d}/{number_of_epochs:03d} "
            f"| MSE: {epoch_loss:.6f}"
        )


# ---------------- Equivariance test ----------------

def random_so3(batch_size):
    matrices = torch.randn(
        batch_size,
        3,
        3,
        device=device,
    )

    q_matrices, r_matrices = torch.linalg.qr(matrices)

    diagonal = torch.diagonal(
        r_matrices,
        dim1=-2,
        dim2=-1,
    )

    signs = torch.sign(diagonal)
    signs[signs == 0] = 1

    q_matrices = q_matrices * signs.unsqueeze(-2)

    determinants = torch.linalg.det(q_matrices)
    q_matrices[determinants < 0, :, 0] *= -1

    return q_matrices


def apply_so3_to_state(rotation_matrices, states):
    q = states[:, 0:3].unsqueeze(-1)
    v = states[:, 3:6].unsqueeze(-1)

    rotated_q = torch.bmm(rotation_matrices, q).squeeze(-1)
    rotated_v = torch.bmm(rotation_matrices, v).squeeze(-1)

    return torch.cat([rotated_q, rotated_v], dim=1)


def empirical_equivariance_so3(model, num_samples, radius):
    model.eval()

    with torch.no_grad():
        times = 2 * math.pi * torch.rand(
            num_samples,
            1,
            device=device,
        )

        states = sample_ball_6d(num_samples, radius)
        rotations = random_so3(num_samples)

        rotated_states = apply_so3_to_state(rotations, states)

        original_inputs = torch.cat([times, states], dim=1)
        rotated_inputs = torch.cat([times, rotated_states], dim=1)

        outputs_on_rotated_states = model(rotated_inputs)
        outputs_on_original_states = model(original_inputs)

        rotated_original_outputs = apply_so3_to_state(
            rotations,
            outputs_on_original_states,
        )

        error = (
            (outputs_on_rotated_states - rotated_original_outputs) ** 2
        ).mean()

    return error.item()


# ---------------- Test-flow error ----------------

def test_flow_error(model, num_samples, radius):
    model.eval()

    with torch.no_grad():
        times = 2 * math.pi * torch.rand(
            num_samples,
            1,
            device=device,
        )

        states = sample_ball_6d(num_samples, radius)

        inputs = torch.cat((times, states), dim=1)
        clean_targets = psi_flow_3d_oscillator(times, states)

        predictions = model(inputs)

        error = ((predictions - clean_targets) ** 2).mean()

    return error.item()


# ---------------- Interactive plots ----------------

def plot_error_surface(
    dataframe,
    fixed_radius,
    error_column,
    output_directory="data/plots",
    use_log10=True,
):
    """
    Plots (depth,width,error) for test and equivariance errors.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Install Plotly to create interactive plots: pip install plotly"
        ) from exc

    selected = dataframe[
        dataframe["R_test"] == fixed_radius
    ].copy()

    error_grid = selected.pivot(
        index="depth",
        columns="width",
        values=error_column,
    )

    error_grid = error_grid.sort_index()
    error_grid = error_grid.sort_index(axis=1)

    depths = error_grid.index.to_numpy()
    widths = error_grid.columns.to_numpy()
    error_values = error_grid.to_numpy(dtype=float)

    if use_log10:
        plotted_values = np.log10(
            np.maximum(error_values, np.finfo(float).tiny)
        )
        z_label = f"log10({error_column})"
    else:
        plotted_values = error_values
        z_label = error_column

    figure = go.Figure(
        data=go.Surface(
            x=widths,
            y=depths,
            z=plotted_values,
        )
    )

    figure.update_layout(
        title=(
            f"{error_column} versus depth and width, "
            f"R_test={fixed_radius}"
        ),
        scene=dict(
            xaxis_title="Width",
            yaxis_title="Depth",
            zaxis_title=z_label,
        ),
    )

    output_path = os.path.join(
        output_directory,
        f"{error_column}_surface_R_{fixed_radius}.html",
    )

    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
    )

    print(f"Saved interactive plot to {output_path}")


def plot_error_vs_parameters(
    dataframe,
    error_column,
    test_radii=(5, 20),
    output_directory="data/plots",
):
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "Install Plotly to create interactive plots: pip install plotly"
        ) from exc

    # Keep only the test radii we want to compare
    selected = dataframe[
        dataframe["R_test"].isin(test_radii)
    ].copy()

    # Convert radius to string so Plotly treats it as a categorical variable
    selected["R_test_label"] = selected["R_test"].astype(str)

    figure = px.scatter(
        selected,
        x="num_parameters",
        y=error_column,
        color="R_test_label",
        color_discrete_map={
            "5": "black",
            "20": "red",
        },
        hover_data=[
            "depth",
            "width",
            "num_parameters",
            "R_test",
        ],
        log_x=True,
        log_y=True,
        title=(
            f"{error_column} versus number of parameters "
            f"for R_test = 5 and 20"
        ),
        labels={
            "num_parameters": "Number of parameters",
            error_column: error_column,
            "R_test_label": "Test radius",
        },
        category_orders={
            "R_test_label": ["5", "20"]
        },
    )

    output_path = os.path.join(
        output_directory,
        f"{error_column}_vs_parameters_R_5_R_20.html",
    )

    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
    )

    print(f"Saved interactive plot to {output_path}")

# ---------------- Experiment parameters ----------------

max_depth = 9
max_width = 36

dataset_size = 10000
training_radius = 5
batch_size = 256
number_of_epochs = 200
noise_std = 0.01

number_of_test_samples = 10000
test_radii = [5, 10, 20, 40]

results_path = "data/empirical_equivariance_vs_depth_width.csv"


# Make the training dataset once and use it for every architecture.
inputs, targets = make_training_data(
    dataset_size,
    training_radius,
    noise_std=noise_std,
)

dataset = torch.utils.data.TensorDataset(inputs, targets)

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)

criterion = nn.MSELoss()


# For each architecture:
#   1. train the model;
#   2. calculate the errors;
#   3. append the errors to the CSV;
#   4. delete the model;
#   5. move to the next architecture.
with open(results_path, "w", newline="") as results_file:
    writer = csv.writer(results_file)

    writer.writerow([
        "depth",
        "width",
        "num_parameters",
        "R_test",
        "eq_error",
        "test_error",
    ])

    results_file.flush()

    for depth in range(1, max_depth + 1):
        for width in range(1, max_width + 1):
            print()
            print(f"Training depth={depth}, width={width}")

            model = NeuralNetwork(depth, width).to(device)

            num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-3,
            )

            train_model(
                model,
                criterion,
                optimizer,
                data_loader,
                number_of_epochs,
            )

            for test_radius in test_radii:
                equivariance_error = empirical_equivariance_so3(
                    model,
                    number_of_test_samples,
                    test_radius,
                )

                flow_error = test_flow_error(
                    model,
                    number_of_test_samples,
                    test_radius,
                )

                writer.writerow([
                    depth,
                    width,
                    num_parameters,
                    test_radius,
                    equivariance_error,
                    flow_error,
                ])

                # Write each row to disk immediately.
                results_file.flush()

                print(
                    f"R_test={test_radius} "
                    f"| eq_error={equivariance_error:.6e} "
                    f"| test_error={flow_error:.6e}"
                )

            del optimizer
            del model

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Generate interactive HTML plots after training is finished.
results = pd.read_csv(results_path)

for fixed_radius in sorted(results["R_test"].unique()):
    plot_error_surface(
        results,
        fixed_radius=fixed_radius,
        error_column="eq_error",
    )

    plot_error_surface(
        results,
        fixed_radius=fixed_radius,
        error_column="test_error",
    )


# Scatter plot of (#parameters, error) where each dot is one (depth,width)
# R_test = 5 in black
# R_test = 20 in red

plot_error_vs_parameters(
    results,
    error_column="eq_error",
    test_radii=(5, 20),
)

plot_error_vs_parameters(
    results,
    error_column="test_error",
    test_radii=(5, 20),
)