import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=r"D:\flow_datasets\harmonic_oscillator_train_data.pt", help="path to dataset .pt file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3) #learning rate
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=512)
    parser.add_argument("--save", type=str, default=r"D:\flow_datasets/checkpoints/model_harm_osc.pth", help="where to save the model state_dict")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    blob = torch.load(args.data, map_location=device)
    inputs = blob["inputs"].to(device)
    targets = blob["targets"].to(device)
    ds = TensorDataset(inputs, targets)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Model, loss, optimizer
    model = build_model(hidden1=args.hidden1, hidden2=args.hidden2, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch_inputs, batch_targets in dl:
            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)
        epoch_loss = running_loss / len(ds)
        print(f"Epoch {epoch:03d} | MSE: {epoch_loss:.6f}")

    # Save weights
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "hidden1": args.hidden1, "hidden2": args.hidden2},
        args.save,
    )
    print(f"Saved model weights to {args.save}")

if __name__ == "__main__":
    main()
