import torch
import torch.nn as nn
from typing import Tuple


class FlowNet(nn.Module):
    """
    Two hidden layers, each: affine (Linear) -> ReLU, followed by a final linear head to R^2.
    Input: (t, q1, q2, p1, p2) ∈ R^5
    Output: (y1, y2, y3, y4) ∈ R^4
    
    
    The "penultimate" layer is the second ReLU activation (high dimensional by design).
    """
    def __init__(self, hidden1: int = 256, hidden2: int = 512):
        super().__init__()
        self.input_dim = 5
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = 4
    
    
        self.fc1 = nn.Linear(self.input_dim, self.hidden1)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.act2 = nn.ReLU() # penultimate representation lives here (dim = hidden2)
        self.out = nn.Linear(self.hidden2, self.output_dim)
    
    
    def features(self, tx: torch.Tensor) -> torch.Tensor:
        """Return the penultimate features (after second ReLU). Shape: (batch, hidden2)."""
        h1 = self.act1(self.fc1(tx))
        h2 = self.act2(self.fc2(h1))
        return h2
    
    
    def forward(self, tx: torch.Tensor, return_penultimate: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        h2 = self.features(tx)
        y = self.out(h2)
        if return_penultimate:
            return y, h2
        return y




def build_model(hidden1: int = 256, hidden2: int = 512, device: str = None) -> FlowNet:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowNet(hidden1=hidden1, hidden2=hidden2).to(device)
    return model