from __future__ import annotations
import random
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    ns: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, size: int = 20000):
        self.buf = deque(maxlen=size)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch: int):
        return random.sample(self.buf, batch)

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3, gamma: float = 0.99):
        self.q = QNet(obs_dim, n_actions)
        self.tgt = QNet(obs_dim, n_actions)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions

    def act(self, obs: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.q(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def update(self, batch: list[Transition]) -> float:
        s = torch.tensor(np.array([b.s for b in batch]), dtype=torch.float32)
        a = torch.tensor([b.a for b in batch], dtype=torch.long)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32)
        ns = torch.tensor(np.array([b.ns for b in batch]), dtype=torch.float32)
        d = torch.tensor([b.done for b in batch], dtype=torch.float32)

        qv = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na = self.q(ns).argmax(dim=1)
            nq = self.tgt(ns).gather(1, na.unsqueeze(1)).squeeze(1)
            y = r + (1 - d) * self.gamma * nq
        loss = nn.functional.mse_loss(qv, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())
