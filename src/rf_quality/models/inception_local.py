"""Self-contained PyTorch InceptionTime-style 1D CNN.

Original implementation for this research project, inspired by the architecture in
Ismail Fawaz et al. (2020), 'InceptionTime: Finding AlexNet for Time Series
Classification'. It is a single residual Inception network, NOT the paper's
five-network ensemble, and not an exact reimplementation of tsai internals.

Only the PyTorch runtime is imported; no tsai or aeon dependency is needed.
"""
from __future__ import annotations

import torch
from torch import nn


class InceptionBlock(nn.Module):
    """Three differently sized temporal convolutions plus a max-pooling branch."""

    def __init__(self, channels: int, filters: int, bottleneck: int = 32):
        super().__init__()
        if channels <= 0 or filters <= 0:
            raise ValueError("channels and filters must be positive")
        bottleneck = min(bottleneck, channels)
        self.reduce = nn.Conv1d(channels, bottleneck, 1, bias=False) if channels > 1 else nn.Identity()
        reduced = bottleneck if channels > 1 else channels
        self.branches = nn.ModuleList([
            nn.Conv1d(reduced, filters, k, padding=k // 2, bias=False)
            for k in (39, 19, 9)
        ])
        self.pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(channels, filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reduce(x)
        joined = torch.cat([layer(z) for layer in self.branches] + [self.pool(x)], dim=1)
        return self.relu(self.bn(joined))


class InceptionNetwork(nn.Module):
    """Six temporal Inception blocks, residual skip every third block, GAP head."""

    def __init__(self, c_in: int = 2, c_out: int = 2, nf: int = 16):
        super().__init__()
        if c_in <= 0 or c_out <= 1 or nf <= 0:
            raise ValueError("Invalid input, output, or filter count")
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        channels = c_in
        for idx in range(6):
            self.blocks.append(InceptionBlock(channels, nf))
            if idx % 3 == 0:
                shortcut_in = channels
            channels = nf * 4
            if idx % 3 == 2:
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(shortcut_in, channels, 1, bias=False), nn.BatchNorm1d(channels)
                ))
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx % 3 == 2:
                x = self.relu(x + self.shortcuts[idx // 3](residual))
                residual = x
        return self.head(self.pool(x).squeeze(-1))
