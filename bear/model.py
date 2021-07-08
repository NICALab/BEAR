# import torch
import torch.nn as nn


class BEAR_WWTY(nn.Module):
    # L = W W^T Y modeling
    def __init__(self, inp, k=1):
        """
        Just contain one linear layer.

        The weight of this layer is the W in the paper.
        """
        super().__init__()
        self.ln = nn.Linear(inp, k, bias=False)

    def forward(self, x):
        L = self.ln(x) @ self.ln.weight
        return L

    def clamper(self):
        for p in self.parameters():
            p.data.clamp_(0.0)


class BEAR_ABY(nn.Module):
    # L = A B Y
    def __init__(self, inp, k=1):
        """
        Use two weight matrix A, B.

        This is relaxed version of BEAR_WWTY,
        which imposes the transpose relationship.

        The weight of this layer is the W in the paper.
        """
        super().__init__()
        self.ln1 = nn.Linear(inp, k, bias=False)
        self.ln2 = nn.Linear(k, inp, bias=False)

    def forward(self, x):
        L = self.ln2(self.ln1(x))
        return L

    def clamper(self):
        for p in self.parameters():
            p.data.clamp_(0.0)
