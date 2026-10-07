"""The one training loop of the deep localizers, plus seeding and batched inference (torch bridge).

Kept free of any localizer logic: ``DeepLocalizer`` standardizes the data (numpy, float64), then calls
``build_network`` and ``train_model`` inside ``seeded(...)``. Everything is deterministic for a fixed
seed and a fixed number of torch threads: parameters are initialised from a private seed, mini-batch
order comes from a numpy ``Generator``, and the global torch RNG and determinism flags are restored
afterwards (the caller's random state is never consumed).
"""
from __future__ import annotations

import contextlib
import copy
import math

import numpy as np
import torch
from torch.nn import functional as F

from .backbones import build_backbone
from .heads import LocalizationNet, MultiTaskHead

__all__ = ["build_network", "predict_outputs", "seeded", "train_model"]


@contextlib.contextmanager
def seeded(seed: int, device: str = "cpu"):
    """Run a block with torch seeded to ``seed`` and deterministic algorithms on; restore both after."""
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    index = (torch.cuda.current_device() if dev.index is None else dev.index) if cuda else None
    with torch.random.fork_rng(devices=[index] if cuda else []):  # forks (and restores) exactly that GPU's RNG
        torch.default_generator.manual_seed(int(seed))
        if cuda:
            with torch.cuda.device(index):  # torch.cuda.manual_seed seeds the *current* device
                torch.cuda.manual_seed(int(seed))
        flags = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])


def build_network(config: dict, *, pretrained: bool = False) -> LocalizationNet:
    """The network described by a DeepLocalizer's ``net_config_`` (a JSON-able dict)."""
    kwargs = dict(config["backbone_kwargs"])
    if config["backbone"] not in ("mlp", "cnn1d"):
        kwargs["pretrained"] = pretrained
    backbone = build_backbone(config["backbone"], config["input_shape"], **kwargs)
    head = MultiTaskHead(backbone.out_features, config["n_outputs"], config["n_floors"], config["n_buildings"],
                         config["head_hidden"], config["head_dropout"])
    return LocalizationNet(backbone, head)


def _batches(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Consecutive slices of ``order``; a trailing batch of one sample joins the previous batch
    (BatchNorm cannot normalise a single sample in training mode)."""
    cuts = list(range(batch_size, len(order), batch_size))
    if cuts and len(order) - cuts[-1] == 1:
        cuts.pop()
    return np.split(order, cuts)


@torch.no_grad()
def predict_outputs(net: torch.nn.Module, X: np.ndarray, *, batch_size: int = 4096,
                    device: str = "cpu") -> dict[str, np.ndarray]:
    """Evaluation-mode forward pass over ``X`` in chunks -> dict of float32 numpy arrays."""
    net.eval()
    parts: dict[str, list] = {}
    for start in range(0, len(X), batch_size):
        chunk = torch.as_tensor(np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32), device=device)
        for key, value in net(chunk).items():
            parts.setdefault(key, []).append(value.float().cpu().numpy())
    return {key: np.concatenate(values) for key, values in parts.items()}


def _mean_distance(net, X, pos, device) -> float:
    pred = predict_outputs(net, X, device=device)["pos"].astype(np.float64)
    return float(np.sqrt(np.square(pred - pos).sum(1)).mean())


def train_model(net: torch.nn.Module, X: np.ndarray, pos: np.ndarray, floor: np.ndarray | None = None,
                building: np.ndarray | None = None, *, validation: tuple | None = None, epochs: int = 100,
                batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-4, patience: int | None = None,
                task_weights=(1.0, 1.0, 1.0), rng: np.random.Generator, device: str = "cpu",
                distance_scale: float = 1.0, verbose: int = 0) -> dict:
    """Mini-batch AdamW on ``w_pos * MSE(pos) + w_floor * CE(floor) + w_building * CE(building)``.

    ``X`` (N, ...) float32 inputs; ``pos`` (N, D) standardized targets; ``floor``/``building`` class
    indices or None. ``validation=(X_val, pos_val)`` (standardized) is scored after every epoch by the
    mean Euclidean distance times ``distance_scale`` (i.e. in coordinate units); with ``patience`` the
    loop stops after that many epochs without improvement and the best weights are restored.
    Returns ``{"loss": [per epoch], "val_error": [per epoch] (with validation only), "best_epoch": int}``
    (``best_epoch`` is the 0-based epoch of the lowest validation error, -1 without validation).
    """
    net.to(device)
    X_t = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=device)
    targets = {"pos": torch.as_tensor(np.ascontiguousarray(pos, dtype=np.float32), device=device)}
    for key, labels in (("floor", floor), ("building", building)):
        if labels is not None:
            targets[key] = torch.as_tensor(labels, dtype=torch.int64, device=device)
    weights = dict(zip(("pos", "floor", "building"), (float(w) for w in task_weights)))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    history: dict[str, list] = {"loss": []}
    best = (math.inf, None, -1)  # (error, state_dict, epoch)
    wait = 0
    if validation is not None:
        history["val_error"] = []
    for epoch in range(epochs):
        net.train()
        total = 0.0
        for idx in _batches(rng.permutation(len(X)), batch_size):
            rows = torch.as_tensor(idx, device=device)
            out = net(X_t[rows])
            loss = weights["pos"] * F.mse_loss(out["pos"], targets["pos"][rows])
            for key in ("floor", "building"):
                if key in targets:
                    loss = loss + weights[key] * F.cross_entropy(out[key], targets[key][rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        history["loss"].append(total / len(X))
        message = f"epoch {epoch + 1}/{epochs}  loss {history['loss'][-1]:.5f}"
        if validation is not None:
            error = _mean_distance(net, *validation, device) * distance_scale
            history["val_error"].append(error)
            message += f"  val_error {error:.4f}"
            if error < best[0]:
                best, wait = (error, copy.deepcopy(net.state_dict()), epoch), 0
            else:
                wait += 1
        if verbose:
            print(message, flush=True)
        if patience is not None and validation is not None and wait >= patience:
            break
    if patience is not None and best[1] is not None:
        net.load_state_dict(best[1])  # early stopping keeps the best epoch, not the last
    net.eval()
    return {**history, "best_epoch": best[2]}
