from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from fewshot import apply_bbox_crop_optimized
from torchvision import models, transforms

try:
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required for metrics. Install with: pip install scikit-learn"
    ) from e

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_label_index_npz(npz_path: str | os.PathLike) -> Dict[int, np.ndarray]:
    """Load your `label_index_*.npz` files where keys are class_ids as strings."""
    z = np.load(npz_path, allow_pickle=True)
    return {int(k): np.asarray(z[k], dtype=int) for k in z.files}


@dataclass(frozen=True)
class SplitIndices:
    train_idx: np.ndarray
    train_y: np.ndarray
    val_idx: np.ndarray
    val_y: np.ndarray
    test_idx: np.ndarray
    test_y: np.ndarray


def _alloc_three_way_counts(n: int, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    """Return (n_train, n_val, n_test) with small-class safeguards."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    if n == 3:
        return 1, 1, 1

    n_val = int(round(val_frac * n))
    n_test = int(round(test_frac * n))
    n_val = max(1, n_val)
    n_test = max(1, n_test)
    n_train = n - n_val - n_test

    if n_train < 2:
        # For fine-grained classification, keep at least 2 train samples when possible.
        deficit = 2 - n_train
        # Reduce val/test (prefer reducing the larger one).
        for _ in range(deficit):
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
        n_train = n - n_val - n_test

    # Final guard: ensure non-negative and sums match.
    n_train = max(1, n_train)
    if n_train + n_val + n_test > n:
        extra = (n_train + n_val + n_test) - n
        # Remove extras from val/test first.
        while extra > 0 and n_test > 0:
            n_test -= 1
            extra -= 1
        while extra > 0 and n_val > 0:
            n_val -= 1
            extra -= 1
        n_train = n - n_val - n_test

    return n_train, n_val, n_test


def split_label_index_train_val_test(
    label_index: Dict[int, np.ndarray],
    class_id_to_idx: Dict[int, int],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> SplitIndices:
    """Stratified 3-way split over the training dataset indices."""
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")
    if not (0.0 <= test_frac < 1.0):
        raise ValueError("test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    train_y: List[int] = []
    val_idx: List[int] = []
    val_y: List[int] = []
    test_idx: List[int] = []
    test_y: List[int] = []

    for class_id in sorted(label_index.keys()):
        idxs = np.asarray(label_index[class_id], dtype=int)
        if idxs.size == 0:
            continue
        rng.shuffle(idxs)

        n_train, n_val, n_test = _alloc_three_way_counts(len(idxs), val_frac, test_frac)

        tr = idxs[:n_train]
        va = idxs[n_train : n_train + n_val]
        te = idxs[n_train + n_val : n_train + n_val + n_test]

        y = int(class_id_to_idx[int(class_id)])
        train_idx.extend(tr.tolist())
        train_y.extend([y] * len(tr))
        val_idx.extend(va.tolist())
        val_y.extend([y] * len(va))
        test_idx.extend(te.tolist())
        test_y.extend([y] * len(te))

    return SplitIndices(
        train_idx=np.asarray(train_idx, dtype=int),
        train_y=np.asarray(train_y, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
        val_y=np.asarray(val_y, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
        test_y=np.asarray(test_y, dtype=int),
    )


class DeepLakeEffNetDataset(Dataset):
    """DeepLake dataset wrapper for EfficientNet-B4 training/eval.

    Preprocess modes:
      - 'native': EfficientNet weights.transforms() (resize + center crop)
      - 'bbox_crop': bbox crop (+padding) -> weights.transforms()
    """

    def __init__(
        self,
        ds,
        indices: np.ndarray,
        y: np.ndarray,
        weights,
        preprocess_mode: str,
        bbox_padding_ratio: float,
        train_aug: bool,
    ):
        self.ds = ds
        self.indices = np.asarray(indices, dtype=int)
        self.y = np.asarray(y, dtype=int)
        self.preprocess_mode = preprocess_mode
        self.bbox_padding_ratio = float(bbox_padding_ratio)

        if self.preprocess_mode not in {"native", "bbox_crop"}:
            raise ValueError("preprocess_mode must be 'native' or 'bbox_crop'")

        base_tf = weights.transforms()
        if train_aug:
            self.tf = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    base_tf,
                ]
            )
        else:
            self.tf = base_tf

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        sample = self.ds[idx]
        img = sample["images"].numpy()
        if self.preprocess_mode == "bbox_crop":
            box = sample["boxes"].numpy()
            img = apply_bbox_crop_optimized(img, box, padding_ratio=self.bbox_padding_ratio)
        x = self.tf(Image.fromarray(img))
        return x, int(self.y[i])


def build_efficientnet_b4(num_classes: int) -> Tuple[nn.Module, object]:
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model, weights


def get_head_module_effnet(model: nn.Module) -> nn.Module:
    return model.classifier


def freeze_backbone_effnet(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in get_head_module_effnet(model).parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def _set_bn_eval(m: nn.Module) -> None:
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def _make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y)
    counts[counts == 0] = 1
    class_w = 1.0 / counts
    sample_w = class_w[y]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool, sampler=None) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _macro_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    n = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    it = loader
    for x, y in it:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    mets = _macro_metrics(y_true, y_pred)
    mets["loss"] = total_loss / max(1, n)
    return mets


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    amp: bool,
    grad_clip_norm: float = 1.0,
):
    model.train()
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None
    total_loss = 0.0
    n = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    it = tqdm(loader, desc="train", leave=False) if tqdm is not None else loader
    for x, y in it:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    mets = _macro_metrics(y_true, y_pred)
    mets["loss"] = total_loss / max(1, n)
    return mets


@torch.no_grad()
def evaluate_with_preds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    topk: int = 5,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    n = 0
    topk_correct = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs

        y_true_batch = y.detach().cpu().tolist()
        y_pred_batch = logits.argmax(dim=1).detach().cpu().tolist()
        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)

        k = int(min(max(1, topk), logits.shape[1]))
        topk_idx = logits.topk(k=k, dim=1).indices
        topk_correct += int((topk_idx == y[:, None]).any(dim=1).sum().item())

    mets = _macro_metrics(y_true, y_pred)
    mets["loss"] = total_loss / max(1, n)
    mets["top5_acc"] = float(topk_correct / max(1, n))
    return mets, y_true, y_pred


@dataclass
class TrainConfig:
    run_name: str = "effnetb4_native_v1"
    preprocess_mode: str = "native"  # native | bbox_crop
    bbox_padding_ratio: float = 0.15
    train_aug: bool = False

    batch_size: int = 16
    val_frac: float = 0.15
    test_frac: float = 0.15

    head_epochs: int = 2
    finetune_epochs: int = 10

    lr_head: float = 3e-3
    lr_backbone: float = 3e-5
    lr_head_finetune: float = 3e-4

    weight_decay: float = 1e-4
    label_smoothing: float = 0.0

    use_weighted_sampler: bool = True
    amp: bool = True
    grad_clip_norm: float = 1.0

    freeze_bn_in_head: bool = True
    freeze_bn_in_finetune: bool = True

    seed: int = 42
    device: str = "auto"
    out_dir: str = "runs"


def train_two_stage_effnetb4(
    cfg: TrainConfig,
    ds_train,
    train_label_index: Dict[int, np.ndarray],
    ds_holdout=None,
    holdout_label_index: Optional[Dict[int, np.ndarray]] = None,
    *,
    splits: Optional[SplitIndices] = None,
    eval_test: bool = True,
    eval_holdout: bool = False,
):
    """Two-stage training: (1) head-only, (2) fine-tune backbone.

    Splits:
      - Train/val/test are created from `train_label_index` (subset of ds_train).
      - `ds_holdout` (original val) is used once at the end as final holdout.
    """
    import pandas as pd

    seed_everything(cfg.seed)
    device = torch.device(pick_device() if cfg.device == "auto" else cfg.device)

    class_ids = sorted(train_label_index.keys())
    class_id_to_idx = {cid: i for i, cid in enumerate(class_ids)}

    if splits is None:
        splits = split_label_index_train_val_test(
            train_label_index,
            class_id_to_idx=class_id_to_idx,
            val_frac=cfg.val_frac,
            test_frac=cfg.test_frac,
            seed=cfg.seed,
        )

    model, weights = build_efficientnet_b4(num_classes=len(class_ids))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    train_ds = DeepLakeEffNetDataset(
        ds_train,
        splits.train_idx,
        splits.train_y,
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        train_aug=cfg.train_aug,
    )
    val_ds = DeepLakeEffNetDataset(
        ds_train,
        splits.val_idx,
        splits.val_y,
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        train_aug=False,
    )
    test_ds = DeepLakeEffNetDataset(
        ds_train,
        splits.test_idx,
        splits.test_y,
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        train_aug=False,
    )

    sampler = _make_weighted_sampler(splits.train_y) if cfg.use_weighted_sampler else None
    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True, sampler=sampler)
    val_loader = _make_loader(val_ds, cfg.batch_size, shuffle=False)
    test_loader = _make_loader(test_ds, cfg.batch_size, shuffle=False)

    holdout_loader = None
    if eval_holdout:
        if ds_holdout is None or holdout_label_index is None:
            raise ValueError("eval_holdout=True requires ds_holdout and holdout_label_index")
        holdout_idx, holdout_y_class_id = [], []
        for class_id, idxs in holdout_label_index.items():
            if class_id not in class_id_to_idx:
                continue
            holdout_idx.extend(np.asarray(idxs, dtype=int).tolist())
            holdout_y_class_id.extend([class_id_to_idx[class_id]] * len(idxs))
        holdout_ds = DeepLakeEffNetDataset(
            ds_holdout,
            np.asarray(holdout_idx, dtype=int),
            np.asarray(holdout_y_class_id, dtype=int),
            weights=weights,
            preprocess_mode=cfg.preprocess_mode,
            bbox_padding_ratio=cfg.bbox_padding_ratio,
            train_aug=False,
        )
        holdout_loader = _make_loader(holdout_ds, cfg.batch_size, shuffle=False)

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    history: List[Dict[str, float]] = []

    def run_stage(
        stage: str,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        freeze_bn: bool,
        global_offset: int,
    ) -> Path:
        best_f1 = -1.0
        best_path = out_dir / f"best_{stage}.pt"

        for ep in range(1, epochs + 1):
            t0 = time.time()
            if freeze_bn:
                model.apply(_set_bn_eval)

            tr = train_one_epoch(
                model,
                train_loader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                amp=cfg.amp,
                grad_clip_norm=cfg.grad_clip_norm,
            )
            va = evaluate(model, val_loader, device=device, criterion=criterion)

            row = {
                "stage": stage,
                "epoch": ep,
                "global_epoch": global_offset + ep,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "train_precision": tr["precision"],
                "train_recall": tr["recall"],
                "train_f1": tr["f1"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
                "val_precision": va["precision"],
                "val_recall": va["recall"],
                "val_f1": va["f1"],
                "seconds": time.time() - t0,
            }
            history.append(row)

            if va["f1"] > best_f1:
                best_f1 = va["f1"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "class_ids": class_ids,
                        "class_id_to_idx": class_id_to_idx,
                        "cfg": asdict(cfg),
                    },
                    best_path,
                )
        return best_path

    # Stage 1: head-only
    freeze_backbone_effnet(model)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    best_head = run_stage("head", cfg.head_epochs, opt1, cfg.freeze_bn_in_head, global_offset=0)
    model.load_state_dict(torch.load(best_head, map_location=device)["model_state"])

    # Stage 2: fine-tune
    unfreeze_all(model)
    head = get_head_module_effnet(model)
    head_params = list(head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head_finetune},
        ],
        weight_decay=cfg.weight_decay,
    )
    best_ft = run_stage(
        "finetune",
        cfg.finetune_epochs,
        opt2,
        cfg.freeze_bn_in_finetune,
        global_offset=cfg.head_epochs,
    )
    model.load_state_dict(torch.load(best_ft, map_location=device)["model_state"])

    test_m = None
    if eval_test:
        test_m, y_true, y_pred = evaluate_with_preds(
            model, test_loader, device=device, criterion=criterion
        )
        (out_dir / "test_classification_report.txt").write_text(
            classification_report(
                y_true,
                y_pred,
                labels=list(range(len(class_ids))),
                target_names=[str(cid) for cid in class_ids],
                digits=3,
                zero_division=0,
            )
        )

    holdout_m = None
    if eval_holdout and holdout_loader is not None:
        holdout_m, y_true, y_pred = evaluate_with_preds(
            model, holdout_loader, device=device, criterion=criterion
        )
        (out_dir / "holdout_classification_report.txt").write_text(
            classification_report(
                y_true,
                y_pred,
                labels=list(range(len(class_ids))),
                target_names=[str(cid) for cid in class_ids],
                digits=3,
                zero_division=0,
            )
        )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)
    summary = {
        "test": test_m,
        "holdout": holdout_m,
        "val_best_f1": float(hist_df["val_f1"].max()) if len(hist_df) else float("nan"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return model, hist_df, summary, out_dir


def plot_training_curves(hist_df) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist_df["global_epoch"], hist_df["train_loss"], label="train_loss")
    axes[0].plot(hist_df["global_epoch"], hist_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(hist_df["global_epoch"], hist_df["train_acc"], label="train_acc")
    axes[1].plot(hist_df["global_epoch"], hist_df["val_acc"], label="val_acc")
    axes[1].plot(hist_df["global_epoch"], hist_df["train_f1"], label="train_f1")
    axes[1].plot(hist_df["global_epoch"], hist_df["val_f1"], label="val_f1")
    axes[1].set_title("Accuracy/F1 (macro)")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
