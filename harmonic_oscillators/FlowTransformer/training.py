import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def autoregressive_loss(model, input_tokens, output_tokens):
    """
    Cross-entropy for the six output coordinates.
    input_tokens:
        shape (batch, 7)
    output_tokens:
        shape (batch, 6)
    Returns:
        scalar cross-entropy averaged over batch and output coordinates.
    """
    # Teacher forcing:
    # [START, Y1, ..., Y5] are the six prediction positions.
    previous_output_tokens = output_tokens[:, :-1]
    # logits shape: (batch, 6, num_bins)
    logits = model(input_tokens, previous_output_tokens)
    # F.cross_entropy expects the class dimension second for an ordinary
    # multidimensional tensor, so flatten batch and coordinate dimensions.
    # Each of the batch * 6 rows is one categorical prediction.
    loss = F.cross_entropy(
        logits.reshape(-1, model.config.num_bins),
        output_tokens.reshape(-1),
        reduction="mean",
    )
    return loss


def train_model(
    model,
    train_data,
    *,
    epochs=20,
    batch_size=256,
    learning_rate=3e-4,
    weight_decay=0.0,
    device=None,
):
    """
    Train FlowTransformer by autoregressive maximum likelihood.

    train_data is a FlowData object returned by make_data().
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    model = model.to(device)

    dataset = TensorDataset(
        train_data.input_tokens.long(),
        train_data.output_tokens.long(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_examples = 0

        for input_tokens, output_tokens in loader:
            input_tokens = input_tokens.to(device)
            output_tokens = output_tokens.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss = autoregressive_loss(
                model,
                input_tokens,
                output_tokens,
            )

            loss.backward()
            optimizer.step()

            batch_size_actual = input_tokens.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual

        epoch_loss = total_loss / total_examples
        loss_history.append(epoch_loss)

        print(
            f"Epoch {epoch:3d}/{epochs} "
            f"| loss = {epoch_loss:.6f}"
        )

    return loss_history