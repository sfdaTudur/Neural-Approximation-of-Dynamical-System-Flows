import csv
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg") #no graphical display if ran on HPC
import matplotlib.pyplot as plt

from data import make_data, dequantize
from model import FlowTransformer, ModelConfig
from training import train_model




TOKENIZATION_RADIUS = 1.0

# Train only on the inner ball of radius R/2.
TRAIN_SAMPLING_RADIUS = TOKENIZATION_RADIUS / 2

# Test on the full ball of radius R
TEST_SAMPLING_RADIUS = TOKENIZATION_RADIUS

NUM_BINS = 256



# Dataset parameters
TRAIN_SIZE = 100_000
TEST_SIZE = 10_000
TRAIN_SEED = 0
TEST_SEED = 1

# Training parameters
EPOCHS = 20
BATCH_SIZE = 256
TEST_BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0


# Architecture parameters
LAYERS = 2
HEADS = 4


RESULTS_DIR = Path("width_results")
RESULTS_FILE = Path("width_vs_mse.csv")
PLOT_FILE = Path("width_vs_mse.png")


@torch.no_grad()
def evaluate_generated_mse(
    model,
    test_data,
    *,
    tokenization_radius,
    num_bins,
    batch_size,
    device,
    is_greedy,
    generation_seed=12345,
):
    """
    Compute test MSE using autoregressive samples from model.generate().

    For every test input:

        1. Generate six output-bin IDs using model.generate().
        2. Replace each generated bin by its midpoint.
        3. Compare the resulting six-dimensional prediction against
           the exact continuous future state.

    Returns
    -------
    mse : float
        Mean squared error averaged over all test examples and all
        six output coordinates.
    """

    model.eval()

    dataset = TensorDataset(
        test_data.input_tokens.long(),
        test_data.future_states,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # model.generate() is stochastic because it uses torch.multinomial.
    # Give it a fixed generator so the experiment is reproducible.
    generator = torch.Generator(device=device)
    generator.manual_seed(generation_seed)

    total_squared_error = 0.0
    total_coordinates = 0

    for input_tokens, true_future_states in loader:
        input_tokens = input_tokens.to(device)
        true_future_states = true_future_states.to(device)

        generated_tokens, _ = model.generate(
            input_tokens,
            generator=generator,
            greedy = is_greedy,
        )

        # Convert each predicted output bin into its midpoint.
        predicted_future_states = dequantize(
            generated_tokens,
            -tokenization_radius,
            tokenization_radius,
            num_bins,
        )

        squared_errors = (
            predicted_future_states - true_future_states
        ).square()

        total_squared_error += squared_errors.sum().item()
        total_coordinates += squared_errors.numel()

    return total_squared_error / total_coordinates


def run_single_width(width):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Width: {width}")

    if width % HEADS != 0:
        raise ValueError(
            f"Width {width} must be divisible by {HEADS} heads."
        )

    # Generate exactly the same datasets for every experiment
    # because the seeds are fixed.
    print("Generating training dataset...")

    train_data = make_data(
        size=TRAIN_SIZE,
        sampling_radius=TRAIN_SAMPLING_RADIUS,
        tokenization_radius=TOKENIZATION_RADIUS,
        num_bins=NUM_BINS,
        seed=TRAIN_SEED,
    )

    print("Generating test dataset...")

    test_data = make_data(
        size=TEST_SIZE,
        sampling_radius=TEST_SAMPLING_RADIUS,
        tokenization_radius=TOKENIZATION_RADIUS,
        num_bins=NUM_BINS,
        seed=TEST_SEED,
    )

    # Reproducible model initialization.
    torch.manual_seed(width)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(width)

    config = ModelConfig(
        num_bins=NUM_BINS,
        width=width,
        layers=LAYERS,
        heads=HEADS,
    )

    model = FlowTransformer(config)

    print(f"Parameters: {model.parameter_count:,}")

    loss_history = train_model(
        model,
        train_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=device,
    )

    print(f"Final training loss: {loss_history[-1]:.6f}")

    mse_stochastic = evaluate_generated_mse(
        model,
        test_data,
        tokenization_radius=TOKENIZATION_RADIUS,
        num_bins=NUM_BINS,
        batch_size=TEST_BATCH_SIZE,
        device=device,
        is_greedy=False,
    )

    mse_greedy = evaluate_generated_mse(
        model,
        test_data,
        tokenization_radius=TOKENIZATION_RADIUS,
        num_bins=NUM_BINS,
        batch_size=TEST_BATCH_SIZE,
        device=device,
        is_greedy=True,
    )

    # Each Slurm task gets its OWN result file.
    RESULTS_DIR.mkdir(exist_ok=True)

    result_file = RESULTS_DIR / f"width_{width:03d}.csv"

    with result_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "width",
            "Stochastic mse",
            "Greedy mse",
        ])

        writer.writerow([
            width,
            mse_stochastic,
            mse_greedy,
        ])

    print(f"Result written to {result_file}")

def aggregate_results():
    """
    Combines data from each experiment into a single csv and plots results
    """
    result_files = list(
        RESULTS_DIR.glob("width_*.csv")
    )

    if not result_files:
        raise RuntimeError(
            f"No result files found in {RESULTS_DIR}"
        )

    results = []

    for result_file in result_files:
        with result_file.open(
            "r",
            encoding="utf-8",
        ) as csv_file:

            reader = csv.DictReader(csv_file)
            row = next(reader)

            results.append({
                "width": int(row["width"]),
                "Stochastic mse": float(
                    row["Stochastic mse"]
                ),
                "Greedy mse": float(
                    row["Greedy mse"]
                ),
            })

    # The jobs may finish in any order,
    # so sort by embedding width.
    results.sort(key=lambda row: row["width"])

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "width",
            "Stochastic mse",
            "Greedy mse",
        ])

        for row in results:
            writer.writerow([
                row["width"],
                row["Stochastic mse"],
                row["Greedy mse"],
            ])

    widths = [
        row["width"]
        for row in results
    ]

    stochastic_mse = [
        row["Stochastic mse"]
        for row in results
    ]

    greedy_mse = [
        row["Greedy mse"]
        for row in results
    ]

    plt.figure(figsize=(7, 5))

    plt.plot(
        widths,
        stochastic_mse,
        marker="o",
        label="Stochastic",
    )

    plt.plot(
        widths,
        greedy_mse,
        marker="o",
        label="Greedy",
    )

    plt.xlabel("Transformer width")
    plt.ylabel("Test MSE")
    plt.title("Transformer Width vs Test MSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()

    print(f"Combined results written to {RESULTS_FILE}")
    print(f"Plot written to {PLOT_FILE}")

def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--width",
        type=int,
        help="Embedding width to train.",
    )

    group.add_argument(
        "--aggregate",
        action="store_true",
        help="Combine results and create the graph.",
    )

    args = parser.parse_args()

    if args.aggregate:
        aggregate_results()
    else:
        run_single_width(args.width)


if __name__ == "__main__":
    main()