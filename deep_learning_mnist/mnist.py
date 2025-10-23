from __future__ import annotations
import os
import json
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch import Tensor
from torch.optim import Optimizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as _random
from typing import (
    Optional,
    Tuple,
    Any,
    Dict,
    Sequence,
    Literal,
    Mapping,
    Callable,
    Iterable,
    List,
    TypedDict,
    Union,
)
from sklearn.metrics import classification_report, confusion_matrix
import math
import optuna

# -------------------------------
# Utilities
# -------------------------------


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all relevant random seeds for reproducibility across Python, NumPy, PyTorch, and PyTorch Lightning.

    This function ensures consistent experimental results by seeding:
    - Python's built-in random module
    - NumPy's random generator
    - PyTorch CPU and CUDA RNGs
    - PyTorch Lightning workers (for DataLoader reproducibility)

    If deterministic is True, it also enables deterministic CuDNN operations,
    which ensures repeatable results on GPU at the cost of possible performance reduction.

    Args:
        seed (int): The random seed value to use. Defaults to 42.
        deterministic (bool): Whether to enforce deterministic GPU behavior.
            If True, sets CuDNN and CUBLAS configs for reproducibility.
            Defaults to True.

    Returns:
        None
    """
    pl.seed_everything(seed, workers=True)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """
    Automatically select the best available computation device.

    This function checks for hardware acceleration options in order of preference:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)

    Returns:
        torch.device: The selected device for model computation.
            - 'cuda' if a CUDA-enabled GPU is available.
            - 'mps' if running on Apple Silicon and MPS is available.
            - 'cpu' otherwise.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------
# Data loading
# -------------------------------


def _seed_worker(worker_id: int) -> None:
    """
    Initialize RNGs inside a DataLoader worker process for reproducible sampling.

    PyTorch's DataLoader may spawn multiple worker processes (via num_workers > 0).
    Each worker needs its own deterministic seed so that NumPy and Python's 'random'
    produce repeatable results (e.g., inside dataset __getitem__ or transforms).

    Behavior:
        - Derives a 32-bit seed from PyTorch's per-worker CUDA/CPU seed (torch.initial_seed()).
        - Seeds NumPy and Python's 'random' with that derived seed.

    Args:
        worker_id (int): Index of the worker process being initialized. (Not used
            directly; included to match the DataLoader API signature.)

    Returns:
        None
    """
    worker_seed: int = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


def make_loaders_train_val(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_mem: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build reproducible DataLoaders for training and validation splits from tensors.

    Args:
        Xtr: Training features tensor of shape (N, 784), dtype float.
        ytr: Training labels tensor of shape (N,), dtype long.
        Xva: Validation features tensor of shape (M, 784), dtype float.
        yva: Validation labels tensor of shape (M,), dtype long.
        batch_size: Batch size for both loaders.
        num_workers: DataLoader worker processes (0 recommended in notebooks).
        pin_mem: Enable pinned memory if True; defaults to True on CUDA.
        seed: Seed for DataLoader shuffling/generator.

    Returns:
        (train_loader, val_loader): Pair of PyTorch DataLoaders.

    Raises:
        AssertionError: If inputs have unexpected shapes or dtypes.
    """
    assert isinstance(Xtr, torch.Tensor) and isinstance(ytr, torch.Tensor)
    assert isinstance(Xva, torch.Tensor) and isinstance(yva, torch.Tensor)
    assert Xtr.ndim == 2 and Xtr.shape[1] == 784, "Xtr must be (N, 784)"
    assert Xva.ndim == 2 and Xva.shape[1] == 784, "Xva must be (N, 784)"
    assert ytr.ndim == 1 and yva.ndim == 1, "labels must be 1D"

    train_ds = TensorDataset(Xtr, ytr)
    val_ds = TensorDataset(Xva, yva)

    if pin_mem is None:
        pin_mem = torch.cuda.is_available()
    persistent = num_workers > 0
    if seed is None:
        seed = 42
    g = torch.Generator().manual_seed(int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    return train_loader, val_loader


def make_loaders_train_val_test(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    Xte: torch.Tensor,
    yte: torch.Tensor,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_mem: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build reproducible DataLoaders for train, val, and test splits from tensors.

    Args:
        Xtr: Training features tensor (N, 784), float.
        ytr: Training labels tensor (N,), long.
        Xva: Validation features tensor (M, 784), float.
        yva: Validation labels tensor (M,), long.
        Xte: Test features tensor (K, 784), float.
        yte: Test labels tensor (K,), long.
        batch_size: Batch size for all loaders.
        num_workers: DataLoader worker processes (0 recommended in notebooks).
        pin_mem: Enable pinned memory if True; defaults to True on CUDA.
        seed: Seed for DataLoader shuffling/generator.

    Returns:
        (train_loader, val_loader, test_loader): Three PyTorch DataLoaders.

    Raises:
        AssertionError: If inputs have unexpected shapes or dtypes.
    """
    assert isinstance(Xtr, torch.Tensor) and isinstance(ytr, torch.Tensor)
    assert isinstance(Xva, torch.Tensor) and isinstance(yva, torch.Tensor)
    assert isinstance(Xte, torch.Tensor) and isinstance(yte, torch.Tensor)
    assert Xtr.ndim == 2 and Xtr.shape[1] == 784, "Xtr must be (N, 784)"
    assert Xva.ndim == 2 and Xva.shape[1] == 784, "Xva must be (N, 784)"
    assert Xte.ndim == 2 and Xte.shape[1] == 784, "Xte must be (N, 784)"
    assert ytr.ndim == 1 and yva.ndim == 1 and yte.ndim == 1, "labels must be 1D"

    train_ds = TensorDataset(Xtr, ytr)
    val_ds = TensorDataset(Xva, yva)
    test_ds = TensorDataset(Xte, yte)
    if pin_mem is None:
        pin_mem = torch.cuda.is_available()

    persistent = num_workers > 0

    if seed is None:
        seed = 42
    g = torch.Generator().manual_seed(int(seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    return train_loader, val_loader, test_loader


# -------------------------------
# Augmented training loader (flat tensors)
# -------------------------------


class AugmentedFlatDataset(Dataset):
    """
    A simple Dataset wrapper that applies a torchvision transform to 28x28 images
    stored as flattened vectors (784,), then flattens them back for an MLP.

    Args:
        X: Float tensor of shape (N, 784) with values typically in [0, 1].
        y: Long tensor of shape (N,) with class labels.
        transform: A torchvision transform that accepts a tensor image (1, 28, 28)
            and returns a tensor image with the same shape.
    """

    def __init__(
        self, X: torch.Tensor, y: torch.Tensor, transform: Optional[Callable] = None
    ) -> None:
        assert X.ndim == 2 and X.shape[1] == 784, "X must be (N, 784)"
        assert y.ndim == 1, "y must be (N,)"
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        img = x.view(1, 28, 28)
        if self.transform is not None:
            img = self.transform(img)
        return img.view(-1), y


def make_augmented_train_loader(
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    transform: Optional[Callable] = None,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_mem: Optional[bool] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Build an augmented training DataLoader for flattened MNIST tensors.

    Args:
        Xtr: Training features tensor (N, 784), float in [0, 1].
        ytr: Training labels tensor (N,), long.
        transform: torchvision-style transform operating on (1, 28, 28) tensors.
        batch_size: Batch size for the training loader.
        num_workers: DataLoader worker processes (0 recommended in notebooks).
        pin_mem: Enable pinned memory if True; defaults to True on CUDA.
        seed: Seed for DataLoader shuffling/generator.

    Returns:
        DataLoader: Augmented training loader suitable for train_loader_override.
    """
    if pin_mem is None:
        pin_mem = torch.cuda.is_available()
    persistent = num_workers > 0
    if seed is None:
        seed = 42
    g = torch.Generator().manual_seed(int(seed))

    ds = AugmentedFlatDataset(Xtr, ytr, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    return loader


# -------------------------------
# Baseline Softmax Model
# -------------------------------


class SoftmaxBaseline(L.LightningModule):
    """
    A minimal softmax (multinomial logistic regression) baseline for 28×28 grayscale images
    flattened to 784 features, predicting one of 10 classes.

    This model is intentionally simple:
      - A single linear layer maps 784 input features to 10 class logits.
      - 'cross_entropy' is used as the loss (it combines 'log_softmax' + NLLLoss).
      - Accuracy is computed by comparing 'argmax' predictions to the labels.

    Args:
        lr (float): Learning rate used by the Adam optimizer. Defaults to 1e-3.
    """

    layer: nn.Linear

    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.layer = nn.Linear(784, 10)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the classifier.

        Expects a batch of flattened images (B, 784) and returns class logits (B, 10).

        Note:
            We do NOT apply softmax here. 'F.cross_entropy' expects raw logits and
            applies 'log_softmax' internally.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 784).

        Returns:
            Tensor: Logits of shape (batch_size, 10).
        """
        return self.layer(x)

    def _step(self, batch: Tuple[Tensor, Tensor], stage: str) -> Tensor:
        """
        Shared train/validation step that computes loss & accuracy and logs them.

        Args:
            batch (Tuple[Tensor, Tensor]): A tuple '(x, y)' where:
                - 'x' has shape (batch_size, 784) with float inputs.
                - 'y' has shape (batch_size,) with integer class indices in [0, 9].
            stage (str): Either '"train"' or '"val"', used to prefix log names.

        Returns:
            Tensor: The scalar loss tensor used for backprop during training.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        One optimization step for the training loop.

        Args:
            batch (Tuple[Tensor, Tensor]): '(x, y)' training batch.
            batch_idx (int): Index of the batch within the current epoch (unused).

        Returns:
            Tensor: The training loss to be minimized.
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        One step of the validation loop (no optimizer step).

        Args:
            batch (Tuple[Tensor, Tensor]): '(x, y)' validation batch.
            batch_idx (int): Index of the batch within the current epoch (unused).
        """
        _ = self._step(batch, "val")

    def configure_optimizers(self) -> Optimizer:
        """
        Create and return the optimizer.

        Returns:
            Optimizer: Adam optimizer over all model parameters using 'self.hparams.lr'.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# -------------------------------
# Multi Layer Perceptron Model
# -------------------------------


class MLP(L.LightningModule):
    """
    A configurable multilayer perceptron (MLP) classifier for flattened image inputs.

    This module builds a stack of fully-connected (Linear) blocks for an input of size
    input_dim and produces logits over 10 classes. Each hidden block is:
        Linear -> (optional BatchNorm1d) -> Activation -> Dropout

    Supported options:
        - Activations: 'relu' | 'leaky' | 'gelu' | 'tanh' | 'elu'
        - Loss:        'ce' (cross-entropy) | 'ce_ls' (CE with label smoothing)
                       | 'focal' (focal loss with gamma/alpha)
        - Optimizers:  'adam' | 'adamw' | 'sgd'
        - Scheduler:   'cosine' (CosineAnnealingLR) | 'none'

    Expected input to forward:
        A float tensor of shape (batch_size, input_dim), i.e., already flattened.
        For 28×28 grayscale images, set input_dim=784.

    Args:
        input_dim (int): Number of input features. Default: 784.
        layers (Sequence[int]): Sizes of hidden layers. Default: (512, 256).
        dropout (float): Dropout probability applied after each activation. Default: 0.3.
        lr (float): Base learning rate. Default: 1e-3.
        opt (Literal['adam','adamw','sgd']): Optimizer choice. Default: 'adamw'.
        weight_decay (float): Weight decay for optimizers that support it. Default: 1e-3.
        use_bn (bool): If True, insert BatchNorm1d after each Linear. Default: True.
        activation (Literal['relu','leaky','gelu','tanh','elu']): Nonlinearity. Default: 'relu'.
        loss (Literal['ce','ce_ls','focal']): Loss function. Default: 'ce'.
        ls (float): Label-smoothing epsilon (used when loss in {'ce','ce_ls'}). Default: 0.0.
        gamma (float): Focal loss γ (how strongly to downweight easy examples). Default: 2.0.
        alpha (Optional[Sequence[float]]): Optional class weights for focal loss.
            If provided, must be length 10 (one weight per class). Default: None.
        scheduler (Literal['cosine','none']): LR scheduler policy. Default: 'cosine'.
        cosine_T_max (int): Number of epochs for CosineAnnealingLR's cycle. Default: 20.
    """

    net: nn.Sequential

    def __init__(
        self,
        input_dim: int = 784,
        layers: Sequence[int] = (512, 256),
        dropout: float = 0.3,
        lr: float = 1e-3,
        opt: Literal["adam", "adamw", "sgd"] = "adamw",
        weight_decay: float = 1e-3,
        use_bn: bool = True,
        activation: Literal["relu", "leaky", "gelu", "tanh", "elu"] = "relu",
        loss: Literal["ce", "ce_ls", "focal"] = "ce",
        ls: float = 0.0,
        gamma: float = 2.0,
        alpha: Optional[Sequence[float]] = None,
        scheduler: Literal["cosine", "none"] = "cosine",
        cosine_T_max: int = 20,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        act_map: Dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "leaky": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }
        if activation not in act_map:
            raise ValueError(f"Unknown activation: {activation!r}")
        Act = act_map[activation]

        if self.hparams.loss == "focal" and alpha is not None and len(alpha) != 10:
            raise ValueError("alpha must be a sequence of length 10 for 10 classes.")

        dims = [input_dim, *layers, 10]
        blocks: list[nn.Module] = []
        for i in range(len(dims) - 2):
            in_f, out_f = dims[i], dims[i + 1]
            blocks.append(nn.Linear(in_f, out_f))
            if use_bn:
                blocks.append(nn.BatchNorm1d(out_f))
            blocks.append(Act())
            blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*blocks)

    # --------------------- Forward & Loss ---------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute class logits.

        Args:
            x (Tensor): Input of shape (batch_size, input_dim), dtype=float.

        Returns:
            Tensor: Logits of shape (batch_size, 10).
        """
        return self.net(x)

    def _compute_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Compute the training/eval loss according to self.hparams.loss.

        Supports:
            - ce / ce_ls: Cross-entropy (with optional label smoothing ls).
            - 'focal': Focal loss with gamma and optional per-class weights alpha.

        Args:
            logits (Tensor): Raw class scores, shape (B, 10).
            y (Tensor): Integer class targets, shape (B,).

        Returns:
            Tensor: Scalar loss tensor.
        """
        loss_name = str(self.hparams.loss).lower()

        if loss_name in ("ce", "ce_ls"):
            return F.cross_entropy(
                logits,
                y,
                label_smoothing=float(self.hparams.ls or 0.0),
            )

        if loss_name == "focal":
            ce_per_sample = F.cross_entropy(logits, y, reduction="none")
            pt = torch.exp(-ce_per_sample)
            loss = ((1.0 - pt) ** float(self.hparams.gamma)) * ce_per_sample
            if self.hparams.alpha is not None:
                a = torch.as_tensor(
                    self.hparams.alpha,
                    device=logits.device,
                    dtype=loss.dtype,
                )
                loss = a[y] * loss

            return loss.mean()

        raise ValueError(f"Unknown loss: {self.hparams.loss!r}")

    # --------------------- Training/Validation ---------------------

    def _step(self, batch: Tuple[Tensor, Tensor], stage: str) -> Tensor:
        """
        Shared logic for train/val steps: forward pass, loss/acc computation, logging.

        Args:
            batch (Tuple[Tensor, Tensor]): (x, y) where x is (B, input_dim),
                y is (B,) with class indices in [0, 9].
            stage (str): Either 'train' or 'val', used to prefix metric names.

        Returns:
            Tensor: The scalar loss tensor (used for backprop during training).
        """
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        One training iteration: compute loss and log metrics.

        Args:
            batch (Tuple[Tensor, Tensor]): Training batch (x, y).
            batch_idx (int): Batch index within the epoch (unused).

        Returns:
            Tensor: Loss for the optimizer to minimize.
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        One validation iteration: compute metrics for monitoring (no optimization).

        Args:
            batch (Tuple[Tensor, Tensor]): Validation batch (x, y).
            batch_idx (int): Batch index within the epoch (unused).
        """
        _ = self._step(batch, "val")

    # --------------------- Optimizer/Scheduler ---------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Create optimizer (Adam/AdamW/SGD) and optional cosine LR scheduler.

        Returns:
            Dict[str, Any]: A Lightning-compatible dictionary containing:
                - optimizer: the created optimizer.
                - lr_scheduler (optional): the LR scheduler, if enabled.
        """
        opt_name = str(self.hparams.opt).lower()
        wd = float(self.hparams.weight_decay or 0.0)

        if opt_name == "adamw":
            opt: Optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=wd
            )
        elif opt_name == "sgd":
            opt = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=wd
            )
        elif opt_name == "adam":
            opt = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=wd
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.opt!r}")

        if str(self.hparams.scheduler).lower() == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=int(self.hparams.cosine_T_max)
            )
            return {"optimizer": opt, "lr_scheduler": sch}

        return {"optimizer": opt}


# -------------------------------
# Run experiment
# -------------------------------
def run_experiment(
    name: str,
    *,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    layers: Sequence[int] = (512, 256),
    dropout: float = 0.3,
    lr: float = 1e-3,
    opt: str = "adamw",
    use_bn: bool = True,
    activation: str = "relu",
    loss: str = "ce",
    ls: float = 0.0,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    weight_decay: float = 1e-4,
    scheduler: str = "cosine",
    cosine_T_max: int = 20,
    batch_size: int = 128,
    max_epochs: int = 25,
    num_workers: int = 2,
    pin_mem: Optional[bool] = None,
    train_loader_override: Optional[DataLoader[Any]] = None,
    log_tensorboard: bool = True,
    save_checkpoints: bool = False,
    save_weights_only: bool = False,
    monitor: str = "val_acc",
    patience: int = 3,
    log_dir: str = "experiments",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train an MLP with PyTorch Lightning and return summary metrics.

    This function wires up:
      - Dataloaders via make_loaders_train_val(...)
      - A Lightning Trainer with EarlyStopping (+ optional ModelCheckpoint)
      - An MLP LightningModule (must exist in your codebase)
      - Optional TensorBoard logging

    Assumptions / Dependencies:
      * You provide make_loaders_train_val(Xtr, ytr, Xva, yva, ...)
        that returns (train_loader, val_loader).
      * You provide an MLP LightningModule with the given constructor signature:
          MLP(layers, dropout, lr, opt, use_bn, activation, loss, ls,
              gamma, alpha, weight_decay, scheduler, cosine_T_max)
      * Your MLP logs at least val_acc and val_loss to be monitored.

    Args:
        name: A short experiment name; used in TensorBoard run directory names.
        layers: Hidden layer sizes for the MLP (e.g., (512, 256)).
        dropout: Dropout probability (0–1).
        lr: Learning rate.
        opt: Optimizer choice string (interpreted in your MLP).
        use_bn: Whether to insert BatchNorm between linear layers.
        activation: Activation function name (interpreted in your MLP).
        loss: Loss function key (interpreted in your MLP, e.g., "ce", "focal").
        ls: Label smoothing factor (if supported by your loss).
        gamma: Focal loss gamma (ignored for non-focal losses).
        alpha: Focal/class weighting factor (optional).
        weight_decay: L2 weight decay for the optimizer.
        scheduler: LR scheduler key (interpreted in your MLP), e.g. "cosine".
        cosine_T_max: Period for CosineAnnealingLR if used.
        batch_size: Training batch size.
        max_epochs: Maximum number of epochs to train.
        num_workers: DataLoader worker processes for background prefetch/CPU transforms.
        pin_mem: If None, set True on CUDA; speeds host→GPU copies. Otherwise use provided.
        train_loader_override: If provided, use this for training instead of the default.
        log_tensorboard: If True, logs to TensorBoard under log_dir/name.
        save_checkpoints: If True, save top-1 checkpoint by monitor.
        save_weights_only: If True, checkpoint excludes optimizer/scheduler states.
        monitor: Metric name to monitor for EarlyStopping/Checkpoint (e.g., "val_acc").
        patience: EarlyStopping patience (epochs without improvement).
        log_dir: Parent directory for experiment logs (TensorBoard).
        seed: Random seed for reproducible runs (weights, dataloader workers, etc).

    Returns:
        A dict with:
            {
                "name":        str,
                "val_acc":     float,     # NaN if not logged
                "val_loss":    float,     # NaN if not logged
                "best_ckpt":   str,       # empty if not saved
                "best_score":  Optional[float],
            }

    Raises:
        ValueError: If the MLP is constructed with unsupported options (e.g., activation).

    Notes:
        * monitor direction is auto-inferred: "min" if "loss" in name, else "max".
        * Set save_checkpoints=True only for runs you want persisted—saves disk space.
    """
    print(f"\n=== {name} ===")
    if pin_mem is None:
        pin_mem = torch.cuda.is_available()

    base_train_loader, val_loader = make_loaders_train_val(
        Xtr=Xtr,
        ytr=ytr,
        Xva=Xva,
        yva=yva,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        seed=seed,
    )
    train_loader: DataLoader[Any] = train_loader_override or base_train_loader

    # --- Early stopping & optional checkpointing ---
    monitor_mode = "min" if "loss" in monitor.lower() else "max"
    callbacks = [EarlyStopping(monitor=monitor, mode=monitor_mode, patience=patience)]

    ckpt: Optional[ModelCheckpoint] = None
    if save_checkpoints:
        ckpt = ModelCheckpoint(
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=1,
            save_last=False,
            filename="epoch-{epoch:04d}-" + monitor + "={" + monitor + ":.4f}",
            save_weights_only=save_weights_only,
        )
        callbacks.insert(0, ckpt)

    logger: bool | TensorBoardLogger
    logger = (
        TensorBoardLogger(save_dir=log_dir, name=name) if log_tensorboard else False
    )

    # --- Accelerator & precision & reproducibility ---
    if torch.cuda.is_available():
        accelerator, precision = "gpu", 16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        accelerator, precision = "mps", 32
    else:
        accelerator, precision = "cpu", 32

    L.seed_everything(seed, workers=True)

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        precision=precision,
        enable_checkpointing=save_checkpoints,
    )

    # --- Model + training ---
    model = MLP(
        layers=layers,
        dropout=dropout,
        lr=lr,
        opt=opt,
        use_bn=use_bn,
        activation=activation,
        loss=loss,
        ls=ls,
        gamma=gamma,
        alpha=alpha,
        weight_decay=weight_decay,
        scheduler=scheduler,
        cosine_T_max=cosine_T_max,
    )

    trainer.fit(model, train_loader, val_loader)

    # --- Collect metrics & checkpoint info & log ---
    metrics = trainer.callback_metrics
    val_acc = float(metrics.get("val_acc", torch.tensor(float("nan"))).item())
    val_loss = float(metrics.get("val_loss", torch.tensor(float("nan"))).item())

    if isinstance(logger, TensorBoardLogger):
        hparams = {
            "layers": layers,
            "dropout": dropout,
            "lr": lr,
            "opt": opt,
            "use_bn": use_bn,
            "activation": activation,
            "loss": loss,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
        }
        results = {"val_acc": val_acc, "val_loss": val_loss}

        logger.log_hyperparams(hparams, results)

    best_path: str = ""
    best_score: Optional[float] = None
    if save_checkpoints and ckpt is not None:
        best_path = ckpt.best_model_path or ""
        if ckpt.best_model_score is not None:
            best_score = float(ckpt.best_model_score)

    print("Best ckpt:", best_path or "(not saved)")
    print("Best metric:", best_score if best_score is not None else val_acc)

    return {
        "name": name,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "best_ckpt": best_path,
        "best_score": best_score,
    }


def make_ablation_grid(
    baseline: Mapping[str, Any],
    *,
    optimizers: Optional[Iterable[str]] = None,
    batch_sizes: Optional[Iterable[int]] = None,
    activations: Optional[Iterable[str]] = None,
    lrs: Optional[Iterable[float]] = None,
    dropouts: Optional[Iterable[float]] = None,
    hidden_archs: Optional[Iterable[Sequence[int]]] = None,
    losses: Optional[Iterable[str]] = None,
    weight_decays: Optional[Iterable[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Build an ablation grid, one factor at a time, from a baseline config.

    This helper takes a single baseline dict of hyperparameters and produces a list of
    configs where exactly one key differs from the baseline in each row. For example:
      - If you pass optimizers=["adam", "sgd"] it will return two configs:
            {baseline..., "optimizer": "adam"}
            {baseline..., "optimizer": "sgd"}
      - If you also pass lrs=[1e-3, 1e-4], it will append two more rows varying only lr.
        In other words, this is not a full Cartesian product; it is an ablation
        where each run isolates a single change versus the baseline.

    Used when you want to attribute performance changes to a single factor.

    Args:
        baseline: Mapping of your default hyperparameters (e.g., {"optimizer": "adam", "lr": 1e-3, ...}).
        optimizers, batch_sizes, activations, lrs, dropouts, hidden_archs, losses, weight_decays:
            Optional iterables of candidate values for that one dimension. If None,
            that dimension is skipped.

    Returns:
        A list of dicts, each starting from baseline with exactly one field overridden.
    """
    key_map: Dict[str, str] = {
        "optimizers": "optimizer",
        "batch_sizes": "batch_size",
        "activations": "activation",
        "lrs": "lr",
        "dropouts": "dropout",
        "hidden_archs": "hidden_layers",
        "losses": "loss",
        "weight_decays": "weight_decay",
    }

    grids: List[Dict[str, Any]] = []

    def add(plural_key: str, values: Iterable[Any]) -> None:
        single_key = key_map[plural_key]
        for v in values:
            d = dict(baseline)
            d[single_key] = v
            grids.append(d)

    if optimizers is not None:
        add("optimizers", optimizers)
    if batch_sizes is not None:
        add("batch_sizes", batch_sizes)
    if activations is not None:
        add("activations", activations)
    if lrs is not None:
        add("lrs", lrs)
    if dropouts is not None:
        add("dropouts", dropouts)
    if hidden_archs is not None:
        add("hidden_archs", hidden_archs)
    if losses is not None:
        add("losses", losses)
    if weight_decays is not None:
        add("weight_decays", weight_decays)

    return grids


def run_sweep(
    prefix: str,
    grid: Sequence[Mapping[str, Any]],
    *,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    tb_logging: bool = True,
    save_ckpt_each: bool = False,
    train_loader_override: Any = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run a sequence of experiments and collect results into a DataFrame.

    This iterates over a list of hyperparameter dicts (e.g., from make_ablation_grid)
    and for each one:
      1) Builds a unique run name like f"{prefix}_exp_001".
      2) Calls your run_experiment(...) function with parameters taken from the dict,
         falling back to reasonable defaults if a key is missing.
      3) Logs results (val_acc, val_loss) and the parameters used into a row.

    Args:
        prefix: A short string that prefixes each run name ("mnist_ablation", etc.).
        grid: A sequence of config dicts. Each dict can include keys like:
              "optimizer", "batch_size", "activation", "lr", "dropout",
              "hidden_layers", "loss", "weight_decay", "max_epochs", "use_bn", etc.
        Xtr, ytr, Xva, yva: Explicit train/val tensors (flattened to (N, 784)).
        tb_logging: If True, enable TensorBoard logging inside run_experiment.
        save_ckpt_each: If True, save a best checkpoint for each run (disk heavy).
        train_loader_override: Optional DataLoader to use instead of the default training loader.

    Returns:
        df, best:
            df   : pandas DataFrame with one row per run, sorted by val_acc (desc) if non-empty.
            best : dict with the best row's fields (empty dict if df is empty).

    Requirements:
        * You must have a run_experiment(...) function available in scope that accepts:
            - name, Xtr, ytr, Xva, yva, layers, dropout, lr, opt, batch_size, max_epochs, use_bn,
              activation, loss, weight_decay, log_tensorboard, save_checkpoints,
              save_weights_only, train_loader_override, patience
          (Adjust here if your signature differs.)
        * run_experiment is expected to return a dict containing at least:
            {"val_acc": float, "val_loss": float, ...}
    """
    rows: List[Dict[str, Any]] = []

    for i, p in enumerate(grid, start=1):
        name = f"{prefix}_exp_{i:03d}"
        res = run_experiment(
            name=name,
            Xtr=Xtr,
            ytr=ytr,
            Xva=Xva,
            yva=yva,
            layers=p.get("hidden_layers", (512, 256)),
            dropout=p.get("dropout", 0.2),
            lr=p.get("lr", 1e-3),
            opt=p.get("optimizer", "adam"),
            batch_size=p.get("batch_size", 128),
            max_epochs=p.get("max_epochs", 8),
            use_bn=p.get("use_bn", True),
            activation=p.get("activation", "relu"),
            loss=p.get("loss", "ce"),
            weight_decay=p.get("weight_decay", 0.0),
            log_tensorboard=tb_logging,
            save_checkpoints=save_ckpt_each,
            save_weights_only=False,
            train_loader_override=train_loader_override,
            patience=3,
        )

        rows.append(
            {
                "name": name,
                "val_acc": res["val_acc"],
                "val_loss": res["val_loss"],
                "p_optimizer": p.get("optimizer"),
                "p_batch_size": p.get("batch_size"),
                "p_activation": p.get("activation"),
                "p_lr": p.get("lr"),
                "p_dropout": p.get("dropout"),
                "p_hidden_layers": p.get("hidden_layers"),
                "p_loss": p.get("loss"),
                "p_weight_decay": p.get("weight_decay"),
                "p_max_epochs": p.get("max_epochs"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("val_acc", ascending=False).reset_index(drop=True)
        best = df.iloc[0].to_dict()
    else:
        best = {}

    return df, best


# -------------------------------
# Model Evaluation
# -------------------------------


@torch.no_grad()
def collect_misclassifications(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device | str] = None,
    limit: Optional[int] = None,
    return_indices: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Run the model over a DataLoader and collect all misclassified examples.

    Each batch is moved to device, predicted with the model, and compared against
    ground-truth labels. Any misclassified samples (features + true/pred labels) are
    gathered across the whole loader and returned as CPU tensors.

    Args:
        model:
            A trained model that maps a batch of inputs to logits. It should accept
            tensors shaped like your dataset (here, flattened images of shape (B, 784)).
        loader:
            A DataLoader yielding (xb, yb) where xb has shape (B, 784) and yb
            contains class indices (dtype long).
        device:
            Which device to use for inference (e.g., "cuda", "mps", or "cpu").
            If None, the model stays on its current device.
        limit:
            Optional cap on how many *samples* to process from the loader. Useful to
            preview results quickly without running over the entire dataset.
        return_indices:
            If True, and the DataLoader yields (xb, yb, idx) triplets (where idx are
            dataset indices), the function will also return those indices for any
            misclassified samples. This can be useful for cross-referencing with the
            original dataset or saving specific examples. If False or if the loader
            does not provide indices, this field is ignored.


    Returns:
        wrong_imgs, wrong_true, wrong_pred, wrong_conf:
            - wrong_imgs: Tensor (N_wrong, 784) flattened inputs for misclassified samples.
            - wrong_true: Tensor (N_wrong,) true labels (long).
            - wrong_pred: Tensor (N_wrong,) predicted labels (long).
            - wrong_conf: Tensor (N_wrong,) model confidence for the predicted class.

            If no misclassifications are found, returns empty CPU tensors with shapes:
                wrong_imgs: (0, 784), wrong_true: (0,), wrong_pred: (0,), wrong_conf: (0,).

    Raises:
        RuntimeError: If the model forward pass fails for a batch.

    Notes:
        * @torch.no_grad() disables gradient tracking for faster, memory-efficient eval.
        * model.eval() switches off dropout/batchnorm updates for deterministic eval.
        * All returned tensors are on CPU so they’re easy to plot with matplotlib/NumPy.
    """
    model.eval()
    if device is not None:
        model.to(device)

    wrong_imgs, wrong_true, wrong_pred, wrong_conf = [], [], [], []
    seen = 0

    for xb, yb in loader:
        if device is not None:
            xb = xb.to(device)

        logits: Tensor = model(xb)
        probs: Tensor = torch.softmax(logits, dim=1)
        preds: Tensor = probs.argmax(dim=1).cpu()
        confs: Tensor = probs.max(dim=1).values.cpu()
        yb_cpu: Tensor = yb.cpu()
        wrong_mask: Tensor = preds != yb_cpu
        if wrong_mask.any():
            mask_dev = wrong_mask.to(xb.device)
            wrong_imgs.append(xb[mask_dev].detach().cpu())
            wrong_true.append(yb_cpu[wrong_mask])
            wrong_pred.append(preds[wrong_mask])
            wrong_conf.append(confs[wrong_mask])

        seen += yb_cpu.shape[0]
        if limit is not None and seen >= limit:
            break

    if wrong_imgs:
        wrong_imgs_t = torch.cat(wrong_imgs, dim=0)
        wrong_true_t = torch.cat(wrong_true, dim=0)
        wrong_pred_t = torch.cat(wrong_pred, dim=0)
        wrong_conf_t = torch.cat(wrong_conf, dim=0)
    else:
        wrong_imgs_t = torch.empty(0, 784)
        wrong_true_t = torch.empty(0, dtype=torch.long)
        wrong_pred_t = torch.empty(0, dtype=torch.long)
        wrong_conf_t = torch.empty(0)
    return wrong_imgs_t, wrong_true_t, wrong_pred_t, wrong_conf_t


def show_grid(
    wrong_imgs: Tensor,
    wrong_true: Tensor,
    wrong_pred: Tensor,
    idx_sel: Optional[Tensor] = None,
    n: int = 64,
    cols: int = 6,
    title: Optional[str] = None,
) -> None:
    """
    Visualize a grid of misclassified examples as 28×28 grayscale images.

    Args:
        wrong_imgs:
            Tensor of shape (N_wrong, 784) holding the *flattened* inputs for the
            misclassified samples (values typically in [0, 1]).
        wrong_true:
            Tensor of shape (N_wrong,) with ground-truth labels for those samples.
        wrong_pred:
            Tensor of shape (N_wrong,) with model-predicted labels for those samples.
        idx_sel:
            Optional 1D LongTensor of indices into the misclassified pool selecting a
            specific subset to show. If None, a random subset (size n) is chosen.
        n:
            Max number of images to show (cap applied after idx_sel if provided).
        cols:
            Number of columns in the grid. The number of rows is computed automatically.
        title:
            Optional figure title.

    Behavior:
        * If there are no misclassifications, prints a message and returns.
        * Images are reshaped from (784,) back to (28, 28) for display.

    Notes:
        * Assumes inputs were originally 28×28 grayscale and flattened.
        * Moves tensors (on CPU) to NumPy implicitly via .numpy().
    """
    if wrong_imgs.numel() == 0:
        print("No misclassifications captured.")
        return

    if idx_sel is None:
        n = min(n, wrong_imgs.shape[0])
        idx_sel = torch.randperm(wrong_imgs.shape[0])[:n]
    else:
        n = min(n, idx_sel.shape[0])
        idx_sel = idx_sel[:n]

    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 2, rows * 2))

    for i, j in enumerate(idx_sel, start=1):
        plt.subplot(rows, cols, i)

        img = wrong_imgs[j].reshape(28, 28).numpy()
        plt.imshow(img, cmap="gray")

        tj = int(wrong_true[j])
        pj = int(wrong_pred[j])
        plt.title(f"T:{tj}  P:{pj}", fontsize=9)
        plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: Optional[torch.device | str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a (trained) model over a DataLoader and return y_true, y_pred, and logits.

    This function:
      1) Puts the model in eval mode and (optionally) moves it to device.
      2) Iterates over loader, runs forward passes, and gathers:
         - true labels (y_true), predicted labels (y_pred), and raw logits.

    Args:
        model:
            A classifier that maps inputs -> logits with shape (B, num_classes).
        loader:
            A torch DataLoader yielding (xb, yb) batches. Typically shuffle=False
            for evaluation to ensure stable/ordered outputs.
        device:
            Device to run inference on ("cuda", "mps", "cpu", or torch.device). If None,
            use the model’s current device.

    Returns:
        y_true:  NumPy array of shape (N,) with ground-truth integer class labels.
        y_pred:  NumPy array of shape (N,) with predicted integer class labels.
        logits:  NumPy array of shape (N, num_classes) with raw (pre-softmax) scores.

    Raises:
        RuntimeError: If the model forward pass fails for a batch.

    Notes:
        * @torch.no_grad() disables gradient tracking → faster + less memory.
        * model.eval() turns off dropout and BN updates for deterministic eval.
        * Tensors are moved to CPU before converting to NumPy for compatibility.
    """
    model.eval()
    if device is not None:
        model.to(device)

    ys, ps, ls = [], [], []
    for xb, yb in loader:
        xb = xb.to(device) if device is not None else xb
        logits_b: Tensor = model(xb)
        ys.append(yb.cpu())
        ps.append(logits_b.argmax(dim=1).cpu())
        ls.append(logits_b.cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    logits = torch.cat(ls).numpy()
    return y_true, y_pred, logits


def report_and_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion (val)",
    normalize: Optional[Literal["true", "pred", "all"]] = "true",
) -> pd.DataFrame:
    """
    Print a classification report and plot a confusion matrix heatmap.

    Args:
        y_true:
            Ground-truth integer labels of shape (N,).
        y_pred:
            Predicted integer labels of shape (N,).
        title:
            Title displayed above the confusion matrix plot.
        normalize:
            Normalization mode for the confusion matrix:
              - "true": rows sum to 1 (per-class recall)
              - "pred": columns sum to 1 (per-class precision)
              - "all":  matrix sums to 1 (overall proportions)
              - None:   raw counts

    Returns:
        rep_df:
            A pandas DataFrame version of sklearn's classification_report
            (per-class precision/recall/f1/support + macro/micro/weighted avg).

    Behavior:
        * Uses zero_division=0 to avoid crashes if a class is never predicted.
        * Heatmap values are shown with 2 decimal places when normalized, or as integers otherwise.
    """
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    plt.figure(figsize=(7, 5))
    fmt = ".2f" if normalize is not None else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return rep_df


# -------------------------------
# Final training and report
# -------------------------------


def load_best_checkpoint_path(
    summary_path: str = "experiments/final_mlp/final_summary.json",
    fallback_ckpt: str = "experiments/final_mlp/logs/version_0/checkpoints/best.ckpt",
    explicit: Optional[str] = None,
) -> str:
    """
    Resolve the best model checkpoint path with a simple precedence order.

    Precedence:
        1) If explicit is provided, return it (after existence check).
        2) Else, try reading summary_path and get "best_ckpt".
        3) Else, fall back to fallback_ckpt.

    Args:
        summary_path: Path to the JSON file saved by final training (contains "best_ckpt").
        fallback_ckpt: Sensible default path to a 'best.ckpt' file if summary is missing.
        explicit: An explicit checkpoint path to use, if supplied.

    Returns:
        The resolved checkpoint path (string).

    Raises:
        FileNotFoundError:
            If no valid checkpoint path can be resolved or the resolved path does not exist.
    """
    if explicit:
        ckpt = explicit
    elif os.path.exists(summary_path):
        with open(summary_path) as f:
            ckpt = json.load(f).get("best_ckpt")
    elif os.path.exists(fallback_ckpt):
        ckpt = fallback_ckpt
    else:
        ckpt = None

    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Best checkpoint not found. Got: {ckpt}")
    return ckpt


# -------------------------------
# Hyperparameter tuning with Optuna
# -------------------------------

ArrayLike = Union[np.ndarray, torch.Tensor]
LayersChoice = Tuple[int, ...]
LAYERS_CHOICES = [(256, 256), (512, 256), (512, 512), (512, 256, 128)]


def objective_aug(
    trial: optuna.trial.Trial,
    Xtr: ArrayLike,
    ytr: ArrayLike,
    Xva: ArrayLike,
    yva: ArrayLike,
    aug: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    """
    Optuna objective function (with **data augmentation**): trains an MLP with a sampled
    hyperparameter set and returns the **validation accuracy**. Differs from objective
    in that it constructs a training DataLoader using the provided augmentation transform
    and passes it via train_loader_override to run_experiment.

    The search space is identical to objective:
      - Optimizer: {"adam", "adamw", "sgd"} with optimizer-specific LR ranges
      - Learning rate:
          * SGD: log-uniform in [1e-2, 1e-1]
          * Adam/AdamW: log-uniform in [3e-4, 3e-3]
      - Batch size: {64, 128, 256}
      - Activation: {"relu", "gelu", "leaky", "tanh", "elu"}
      - Dropout: uniform in [0.0, 0.5] with step 0.1
      - Hidden layer architecture: index into LAYERS_CHOICES
      - Loss: {"ce", "ce_ls", "focal"} with ls or gamma as applicable

    Notes
    -----
    - Assumes:
        * make_augmented_train_loader(X, y, transform, batch_size, num_workers, seed) -> DataLoader
        * run_experiment(...) -> Dict[str, Any] with "val_acc" key.
    - aug should be a callable that maps a single image tensor (or batch) to an
      augmented version; e.g., a torchvision.transforms.Compose.
    - The overridden training loader is used **only** for training; validation still
      uses a clean (non-augmented) loader built inside run_experiment.

    Parameters
    ----------
    trial
        Optuna trial object used to sample hyperparameters.
    Xtr, ytr
        Training features and labels (before augmentation).
    Xva, yva
        Validation features and labels.
    aug
        A callable transform applied within the custom training DataLoader.

    Returns
    -------
    float
        Validation accuracy for the sampled hyperparameters with augmentation enabled.
        Optuna should be configured to **maximize** this objective.
    """
    opt: str = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    if opt == "sgd":
        lr: float = trial.suggest_float("lr", 3e-2, 3e-1, log=True)
    else:
        lr = trial.suggest_float("lr", 3e-4, 1e-2, log=True)

    batch_size: int = trial.suggest_categorical("batch_size", [64, 128, 256])
    activation: str = trial.suggest_categorical(
        "activation", ["relu", "gelu", "leaky", "tanh", "elu"]
    )
    dropout: float = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)

    layers_idx: int = trial.suggest_categorical(
        "layers_idx", list(range(len(LAYERS_CHOICES)))
    )
    layers: LayersChoice = LAYERS_CHOICES[layers_idx]

    loss: str = trial.suggest_categorical("loss", ["ce", "ce_ls", "focal"])
    ls: float = trial.suggest_float("ls", 0.05, 0.2) if loss == "ce_ls" else 0.0
    gamma: float = trial.suggest_float("gamma", 1.0, 3.0) if loss == "focal" else 2.0
    use_weight_decay: bool = trial.suggest_categorical(
        "use_weight_decay", [False, True]
    )
    weight_decay_val: float = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    weight_decay: float = float(weight_decay_val) if use_weight_decay else 0.0

    train_aug_loader: DataLoader = make_augmented_train_loader(
        Xtr,
        ytr,
        transform=aug,
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )

    res: Dict[str, Any] = run_experiment(
        name=f"optuna_aug_{trial.number}",
        Xtr=Xtr,
        ytr=ytr,
        Xva=Xva,
        yva=yva,
        layers=layers,
        dropout=dropout,
        lr=lr,
        opt=opt,
        use_bn=True,
        activation=activation,
        loss=loss,
        ls=ls,
        gamma=gamma,
        alpha=None,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=10,
        patience=2,
        num_workers=0,
        log_tensorboard=False,
        save_checkpoints=False,
        train_loader_override=train_aug_loader,
    )
    return float(res["val_acc"])
