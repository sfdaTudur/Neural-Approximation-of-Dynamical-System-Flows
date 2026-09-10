import csv
import gc

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

# ModelConfig requires width % heads == 0.
WIDTHS = [
    width
    for width in range(1, 71)
    if width % HEADS == 0
]


RESULTS_FILE = "width_vs_mse.csv"


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


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Widths: {WIDTHS}")
    print()

    # Generate the datasets once; every architecture sees the same
    # training examples and test examples.

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

    print(f"Training examples: {len(train_data):,}")
    print(f"Test examples:     {len(test_data):,}")
    print()

    width_results = []
    mse_results_stochastic = []
    mse_results_greedy = []

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "width",
                "Stochastic mse",
                "Greedy mse",
            ]
        )

        for width in WIDTHS:

            print("=" * 60)
            print(f"Training transformer with width = {width}")
            print("=" * 60)

            # Give each experiment a reproducible initialization.
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

            print(
                f"Parameters: "
                f"{model.parameter_count:,}"
            )

            #train
            loss_history = train_model(
                model,
                train_data,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                device=device,
            )

            model.eval()

            print(
                f"Final training loss: "
                f"{loss_history[-1]:.6f}"
            )

            # Test using model.generate().
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

            width_results.append(width)
            mse_results_stochastic.append(mse_stochastic)
            mse_results_greedy.append(mse_greedy)


            # Save result before destroying model.
            writer.writerow(
                [
                    width,
                    mse_stochastic,
                    mse_greedy
                ]
            )

            # Force the CSV contents to disk after every model.
            csv_file.flush()

            # Delete this model before constructing the next.
            del model
            del config
            del loss_history

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print()
    #plot (width,mse) for stochastic and greedy generation.
    plt.figure(figsize=(7, 5))
    plt.plot(
        width_results,
        mse_results_stochastic,
        marker="o",
        label ="Stochastic",
    )
    plt.plot(
        width_results,
        mse_results_greedy,
        marker="o",
        label="Greedy",
    )

    plt.xlabel("Transformer width")
    plt.ylabel("Test MSE")
    plt.title("Transformer Width vs Test MSE")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        "width_vs_mse.png",
        dpi=300,
    )

    plt.close()


    print("=" * 60)
    print("Experiment complete.")
    print(f"Results written to: {RESULTS_FILE}")
    print("Plot written to: width_vs_mse.png")
    print("=" * 60)


if __name__ == "__main__":
    main()