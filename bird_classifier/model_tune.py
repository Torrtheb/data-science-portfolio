from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Mapping
from pathlib import Path
import json
import os
import sys
import random
import time
from collections import defaultdict

# Data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
from types import SimpleNamespace

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
from torchvision import transforms as T
from torchvision import transforms

# Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Image processing
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset
import deeplake

# =============================================================================
# MODULE-LEVEL CONSTANTS
# =============================================================================

def get_device() -> torch.device:
    """Pick the best available device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline config - can be overridden when calling train functions
BASELINE_CONFIG = dict(
    batch_size=32,
    head_epochs=3,
    finetune_epochs=10,
    lr_head=3e-3,
    lr_backbone=3e-5,
    lr_head_finetune=3e-4,
    weight_decay=1e-4,
    label_smoothing=0.0,
    use_weighted_sampler=True,
    use_amp=True,
    use_torch_compile=True,
    grad_clip_norm=1.0,
    freeze_bn_head=True,
    freeze_bn_finetune=True,
    augmentation="none",
    augmentation_params=None,
    resume_head_ckpt=None,
    bbox_padding_ratio=0.15,
    pad_to_square=True,
    cache_dir=None,
    cache_version="v1_uint8_rgb_pad_bbox",
    early_stop_patience=3,  # Stop if no improvement for 3 epochs (0 = disabled)
    seed=42,
)

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# LABEL INDEX UTILITIES
# =============================================================================

def build_label_index(ds) -> Dict[int, np.ndarray]:
    """
    Build a mapping: class_label → array of dataset indices.
    
    Why this is useful:
    - Enables stratified splitting (same proportion of each class)
    - Required for few-shot sampling (N images per class)
    - Faster than iterating through dataset multiple times
    """
    label_to_idxs: Dict[int, list] = defaultdict(list)
    for i, sample in tqdm(enumerate(ds), total=len(ds), desc="Building label index"):
        label = int(sample["labels"].numpy()[0])
        label_to_idxs[label].append(int(i))
    return {k: np.array(v, dtype=np.int64) for k, v in label_to_idxs.items()}


def save_label_index(label_index: Dict[int, np.ndarray], path) -> None:
    """Save label index to compressed numpy file."""
    np.savez_compressed(path, **{str(k): v for k, v in label_index.items()})


def load_label_index(path) -> Dict[int, np.ndarray]:
    """Load label index from numpy file."""
    data = np.load(path)
    return {int(k): data[k] for k in data.files}


# =============================================================================
# EXPERIMENT PERSISTENCE (for resuming across Colab sessions)
# =============================================================================

def check_experiment_exists(run_name: str, runs_dir: Path = None) -> bool:
    """
    Check if a completed experiment exists with all required files.
    
    Returns True if the experiment directory contains:
    - config.json
    - history.csv
    - summary.json
    - At least one checkpoint (best_head.pt or best_finetune.pt)
    """
    runs_dir = runs_dir or RUNS_DIR
    run_dir = runs_dir / run_name
    
    if not run_dir.exists():
        return False
    
    required_files = ["config.json", "history.csv", "summary.json"]
    for f in required_files:
        if not (run_dir / f).exists():
            return False
    
    # At least one checkpoint must exist
    has_checkpoint = (run_dir / "best_head.pt").exists() or (run_dir / "best_finetune.pt").exists()
    return has_checkpoint


def load_experiment_results(run_name: str, runs_dir: Path = None) -> Optional[Dict[str, Any]]:
    """
    Load previously saved experiment results (config, history, summary).
    
    Returns None if experiment doesn't exist or is incomplete.
    Returns dict with keys: config, history_df, summary, run_dir
    """
    runs_dir = runs_dir or RUNS_DIR
    run_dir = runs_dir / run_name
    
    if not check_experiment_exists(run_name, runs_dir):
        return None
    
    try:
        config = json.loads((run_dir / "config.json").read_text())
        history_df = pd.read_csv(run_dir / "history.csv")
        summary = json.loads((run_dir / "summary.json").read_text())
        
        return {
            "config": config,
            "history_df": history_df,
            "summary": summary,
            "run_dir": run_dir,
        }
    except Exception as e:
        print(f"Warning: Failed to load experiment '{run_name}': {e}")
        return None


def export_experiments_for_github(
    run_names: List[str],
    output_dir: Path,
    runs_dir: Path = None,
    include_checkpoints: bool = False,
) -> Dict[str, bool]:
    """
    Export experiment results to a directory suitable for GitHub.
    
    This exports lightweight files (config, history, summary) that can be
    committed to GitHub and downloaded in Colab to skip re-training.
    
    Args:
        run_names: List of experiment names to export
        output_dir: Directory to save exported files
        runs_dir: Source runs directory (default: RUNS_DIR)
        include_checkpoints: If True, also copy model checkpoints (large files!)
    
    Returns:
        Dict mapping run_name -> success status
    """
    import shutil
    
    runs_dir = runs_dir or RUNS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    status = {}
    lightweight_files = ["config.json", "history.csv", "summary.json", 
                         "test_classification_report.txt", "holdout_classification_report.txt"]
    checkpoint_files = ["best_head.pt", "best_finetune.pt"]
    
    for run_name in run_names:
        run_dir = runs_dir / run_name
        out_run_dir = output_dir / run_name
        
        if not run_dir.exists():
            print(f"  Skipping '{run_name}': not found")
            status[run_name] = False
            continue
        
        out_run_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy lightweight files
            for f in lightweight_files:
                src = run_dir / f
                if src.exists():
                    shutil.copy2(src, out_run_dir / f)
            
            # Optionally copy checkpoints
            if include_checkpoints:
                for f in checkpoint_files:
                    src = run_dir / f
                    if src.exists():
                        shutil.copy2(src, out_run_dir / f)
            
            status[run_name] = True
            print(f"  Exported '{run_name}'")
        except Exception as e:
            print(f"  Failed '{run_name}': {e}")
            status[run_name] = False
    
    return status


def download_experiments_from_github(
    github_raw_base_url: str,
    run_names: List[str],
    runs_dir: Path = None,
    force: bool = False,
) -> Dict[str, bool]:
    """
    Download experiment results from GitHub raw URLs.
    
    This allows resuming experiments in Colab without re-training by
    downloading previously saved results from GitHub.
    
    Args:
        github_raw_base_url: Base URL for raw GitHub files 
            (e.g., 'https://raw.githubusercontent.com/user/repo/branch/bird_classifier/runs')
        run_names: List of experiment names to download
        runs_dir: Local runs directory (default: RUNS_DIR)
        force: If True, overwrite existing files
    
    Returns:
        Dict mapping run_name -> success status
    """
    import urllib.request
    import urllib.error
    
    runs_dir = runs_dir or RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_download = ["config.json", "history.csv", "summary.json"]
    status = {}
    
    for run_name in run_names:
        run_dir = runs_dir / run_name
        
        # Skip if already exists and not forcing
        if not force and check_experiment_exists(run_name, runs_dir):
            print(f"  '{run_name}': already exists (use force=True to overwrite)")
            status[run_name] = True
            continue
        
        run_dir.mkdir(parents=True, exist_ok=True)
        success = True
        
        for filename in files_to_download:
            url = f"{github_raw_base_url}/{run_name}/{filename}"
            dest = run_dir / filename
            
            try:
                urllib.request.urlretrieve(url, str(dest))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # File not found - might be optional (like classification reports)
                    if filename in ["config.json", "history.csv", "summary.json"]:
                        success = False
                        print(f"  '{run_name}': missing required file {filename}")
                else:
                    success = False
                    print(f"  '{run_name}': HTTP error downloading {filename}: {e}")
            except Exception as e:
                success = False
                print(f"  '{run_name}': error downloading {filename}: {e}")
        
        if success:
            print(f"  Downloaded '{run_name}'")
        status[run_name] = success
    
    return status


def get_completed_experiments(runs_dir: Path = None) -> List[str]:
    """
    Get list of all completed experiment names in the runs directory.
    """
    runs_dir = runs_dir or RUNS_DIR
    if not runs_dir.exists():
        return []
    
    completed = []
    for run_dir in runs_dir.iterdir():
        if run_dir.is_dir() and check_experiment_exists(run_dir.name, runs_dir):
            completed.append(run_dir.name)
    
    return sorted(completed)


def build_results_dataframe(run_names: List[str] = None, runs_dir: Path = None) -> pd.DataFrame:
    """
    Build a results DataFrame from saved experiment files.
    
    This reconstructs the runner.df() output from saved files, useful when
    resuming a notebook session.
    
    Args:
        run_names: List of experiment names (default: all completed experiments)
        runs_dir: Runs directory (default: RUNS_DIR)
    
    Returns:
        DataFrame with experiment results, similar to runner.df()
    """
    runs_dir = runs_dir or RUNS_DIR
    
    if run_names is None:
        run_names = get_completed_experiments(runs_dir)
    
    rows = []
    for run_name in run_names:
        result = load_experiment_results(run_name, runs_dir)
        if result is None:
            continue
        
        config = result["config"]
        summary = result["summary"]
        test = summary.get("test") or {}
        
        row = {
            "run_name": run_name,
            "preprocess_mode": config.get("preprocess_mode"),
            "augmentation": config.get("augmentation"),
            "val_best_f1": summary.get("val_best_f1"),
            "test_f1": test.get("f1"),
            "test_precision": test.get("precision"),
            "test_recall": test.get("recall"),
            "test_acc": test.get("acc"),
            "test_top5_acc": test.get("top5_acc"),
            "test_weighted_f1": test.get("weighted_f1"),
            "test_weighted_precision": test.get("weighted_precision"),
            "test_weighted_recall": test.get("weighted_recall"),
            "run_dir": str(result["run_dir"]),
        }
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows).sort_values("val_best_f1", ascending=False)


# =============================================================================
# DATA SPLITTING
# =============================================================================

@dataclass(frozen=True)
class SplitIndices:
    """Container for train/val/test split indices and labels."""
    train_idx: np.ndarray
    train_y: np.ndarray
    val_idx: np.ndarray
    val_y: np.ndarray
    test_idx: np.ndarray
    test_y: np.ndarray
    
    def __repr__(self):
        return (f"SplitIndices(train={len(self.train_idx)}, "
                f"val={len(self.val_idx)}, test={len(self.test_idx)})")


def _allocate_split_counts(n: int, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    """
    Calculate how many samples go to train/val/test for a single class.
    
    Handles edge cases for small classes:
    - 1 sample → all to train
    - 2 samples → 1 train, 1 val
    - 3 samples → 1 each
    - More → respect fractions with min 1 for val/test, min 2 for train
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    if n == 3:
        return 1, 1, 1

    n_val = max(1, int(round(val_frac * n)))
    n_test = max(1, int(round(test_frac * n)))
    n_train = n - n_val - n_test

    # Ensure at least 2 training samples when possible
    if n_train < 2:
        deficit = 2 - n_train
        for _ in range(deficit):
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
        n_train = n - n_val - n_test

    return max(1, n_train), n_val, n_test


def split_label_index_stratified(
    label_index: Dict[int, np.ndarray],
    class_id_to_idx: Dict[int, int],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """
    Stratified 3-way split of dataset indices.
    
    Args:
        label_index: Dict mapping class_id → array of dataset indices
        class_id_to_idx: Dict mapping class_id → contiguous label (0, 1, 2, ...)
        val_frac: Fraction for validation (default 0.15)
        test_frac: Fraction for test (default 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        SplitIndices with train/val/test indices and corresponding labels
    
    Note:
        - Each class gets approximately the same train/val/test ratio
        - Small classes are handled specially to ensure minimum representation
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError("val_frac must be in (0, 1)")
    if not (0.0 <= test_frac < 1.0):
        raise ValueError("test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    rng = np.random.default_rng(seed)

    train_idx, train_y = [], []
    val_idx, val_y = [], []
    test_idx, test_y = [], []

    for class_id in sorted(label_index.keys()):
        idxs = np.asarray(label_index[class_id], dtype=int)
        if idxs.size == 0:
            continue
        rng.shuffle(idxs)

        n_train, n_val, n_test = _allocate_split_counts(len(idxs), val_frac, test_frac)
        
        y = class_id_to_idx[int(class_id)]

        train_idx.extend(idxs[:n_train].tolist())
        train_y.extend([y] * n_train)
        
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        val_y.extend([y] * n_val)
        
        test_idx.extend(idxs[n_train + n_val:n_train + n_val + n_test].tolist())
        test_y.extend([y] * n_test)

    return SplitIndices(
        train_idx=np.asarray(train_idx, dtype=int),
        train_y=np.asarray(train_y, dtype=int),
        val_idx=np.asarray(val_idx, dtype=int),
        val_y=np.asarray(val_y, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
        test_y=np.asarray(test_y, dtype=int),
    )



# =============================================================================
# IMAGE PREPROCESSING UTILITIES
# =============================================================================

def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert any image format to uint8 RGB.
    
    Handles:
    - Float images (0-1 or 0-255)
    - Grayscale images (2D or with 1 channel)
    - RGBA images (4 channels)
    """
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        max_val = float(arr.max()) if arr.size else 1.0
        if max_val <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unexpected image shape: {arr.shape}")

    return np.ascontiguousarray(arr)


def _pad_to_square(img: np.ndarray) -> np.ndarray:
    """
    Pad image to square using reflection padding.
    
    This preserves aspect ratio and avoids black borders which could
    confuse the model.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    if h > w:
        pad = h - w
        left = pad // 2
        right = pad - left
        top = bottom = 0
    else:
        pad = w - h
        top = pad // 2
        bottom = pad - top
        left = right = 0
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)


# =============================================================================
# BOUNDING BOX UTILITIES (OPTIMIZED)
# =============================================================================

def resolve_bbox_from_box_array(
    box: np.ndarray, 
    img_h: int, 
    img_w: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert bbox to pixel-space (x1, y1, x2, y2) clipped to image bounds.
    
    OPTIMIZED: Takes pre-loaded box array and image dimensions to avoid
    redundant image loading. Use this instead of resolve_bbox_xywh_or_xyxy()
    when the image is already loaded.
    
    Handles:
    - Normalized xyxy (0-1 range)
    - xywh format
    - xyxy format
    
    Args:
        box: Bounding box array from dataset (shape [4,] or [1, 4])
        img_h: Image height in pixels
        img_w: Image width in pixels
    
    Returns:
        (x1, y1, x2, y2) in pixel coordinates, or None if invalid
    """
    box = np.asarray(box, dtype=float).squeeze()
    if box.shape[-1] != 4:
        return None
    
    x1, y1, x2, y2 = box
    h, w = img_h, img_w
    
    # Normalized corners (0-1 range)
    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
    else:
        # (x, y, width, height) format
        width, height = x2, y2
        if width > 0 and height > 0 and x1 + width <= w + 1e-3 and y1 + height <= h + 1e-3:
            x2 = x1 + width
            y2 = y1 + height
        # else assume already (x1, y1, x2, y2)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def resolve_bbox_xywh_or_xyxy(ds, idx: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Legacy wrapper - loads image to get dimensions.
    DEPRECATED: Prefer resolve_bbox_from_box_array() when image is already loaded.
    
    This function loads the image from the dataset, which is wasteful if
    the image is already loaded elsewhere. Use resolve_bbox_from_box_array()
    with the image dimensions instead.
    """
    try:
        img = ds["images"][idx].numpy()
        h, w = img.shape[:2]
        box = ds["boxes"][idx].numpy()
    except Exception:
        return None
    return resolve_bbox_from_box_array(box, h, w)


def apply_bbox_crop_optimized(
    img: np.ndarray, 
    box: np.ndarray, 
    padding_ratio: float = 0.15
) -> np.ndarray:
    """
    OPTIMIZED: Crop to bounding box with padding.
    Takes pre-loaded image and box array to avoid redundant DeepLake access.
    
    Args:
        img: Input image (H, W, C) as numpy array
        box: Bounding box array from dataset
        padding_ratio: Extra padding around bbox as fraction of bbox size (default 15%)
    
    Returns:
        Cropped image. If bbox is invalid, returns original image.
    """
    h, w = img.shape[:2]
    bbox = resolve_bbox_from_box_array(box, h, w)
    if bbox is None:
        return img

    x1, y1, x2, y2 = map(int, bbox)

    box_w, box_h = x2 - x1, y2 - y1
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    # Calculate desired crop region (may extend beyond image)
    crop_x1 = x1 - pad_x
    crop_y1 = y1 - pad_y
    crop_x2 = x2 + pad_x
    crop_y2 = y2 + pad_y

    # Calculate how much padding we need on each side
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)

    # Clip crop region to valid image bounds
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x2)
    crop_y2 = min(h, crop_y2)

    # Guard against degenerate boxes
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return img

    # Crop first
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # Add reflection padding if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101
        )

    return cropped


def apply_bbox_crop(img: np.ndarray, ds, idx: int, padding_ratio: float = 0.15) -> np.ndarray:
    """
    Legacy wrapper for backward compatibility.
    DEPRECATED: Use apply_bbox_crop_optimized() with pre-loaded box array.
    
    Note: This still loads the box from ds, but avoids double image loading
    since img is already passed in.
    """
    try:
        box = ds["boxes"][idx].numpy()
    except Exception:
        return img
    return apply_bbox_crop_optimized(img, box, padding_ratio)


# =============================================================================
# AUGMENTATION REGISTRY
# =============================================================================

# Registry pattern: allows easy addition of new augmentation pipelines
# for hyperparameter tuning experiments.

AUGMENTATION_REGISTRY: Dict[str, Callable[[], A.Compose]] = {}


def register_augmentation(name: str):
    """Decorator to register an augmentation pipeline."""
    def decorator(func):
        AUGMENTATION_REGISTRY[name] = func
        return func
    return decorator


def get_augmentation(name: str) -> Optional[A.Compose]:
    """Get an augmentation pipeline by name. Returns None if 'none'."""
    if name == "none" or name is None:
        return None
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}. Available: {list(AUGMENTATION_REGISTRY.keys())}")
    return AUGMENTATION_REGISTRY[name]()

class BackgroundSubtract(A.ImageOnlyTransform):
    def __init__(
        self,
        *,
        sigma: float = 3.0,
        strength: float = 0.8,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.sigma = float(sigma)
        self.strength = float(strength)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        sigma = max(self.sigma, 0.0)
        if sigma <= 0:
            return img

        ksize = int(2 * round(3 * sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(ksize, 3)

        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        out = img.astype(np.float32) - self.strength * blurred.astype(np.float32) + 128.0
        return np.clip(out, 0, 255).astype(np.uint8)


def build_recipe_augmentation(params: Mapping[str, Any]) -> Optional[A.Compose]:
    """Build a parameterized augmentation recipe (Optuna-friendly)."""
    if not params:
        return None

    def _p(key: str, default: float = 0.0) -> float:
        try:
            return float(params.get(key, default) or 0.0)
        except Exception:
            return default

    ops: List[Any] = []

    p_flip = _p("p_flip", 0.0)
    if p_flip > 0:
        ops.append(A.HorizontalFlip(p=min(max(p_flip, 0.0), 1.0)))

    p_affine = _p("p_affine", 0.0)
    if p_affine > 0:
        shift_limit = float(params.get("shift_limit", 0.0))
        scale_limit = float(params.get("scale_limit", 0.0))
        rotate_limit = float(params.get("rotate_limit", 0.0))
        ops.append(
            A.ShiftScaleRotate(
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                border_mode=cv2.BORDER_REFLECT_101,
                p=min(max(p_affine, 0.0), 1.0),
            )
        )

    p_color = _p("p_color", 0.0)
    if p_color > 0:
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        saturation = float(params.get("saturation", 0.0))
        hue = float(params.get("hue", 0.0))
        ops.append(
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=min(max(p_color, 0.0), 1.0),
            )
        )

    p_clahe = _p("p_clahe", 0.0)
    if p_clahe > 0:
        clip_limit = float(params.get("clahe_clip_limit", 2.0))
        tile_grid_size = int(params.get("clahe_tile_grid_size", 8))
        ops.append(
            A.CLAHE(
                clip_limit=clip_limit,
                tile_grid_size=(tile_grid_size, tile_grid_size),
                p=min(max(p_clahe, 0.0), 1.0),
            )
        )

    p_bgsub = _p("p_bg_subtract", 0.0)
    if p_bgsub > 0:
        sigma = float(params.get("bg_sigma", 3.0))
        strength = float(params.get("bg_strength", 0.8))
        ops.append(
            BackgroundSubtract(
                sigma=sigma,
                strength=strength,
                p=min(max(p_bgsub, 0.0), 1.0),
            )
        )

    p_blur = _p("p_blur", 0.0)
    if p_blur > 0:
        blur_max = int(params.get("blur_max", 5))
        blur_max = max(3, blur_max)
        if blur_max % 2 == 0:
            blur_max += 1
        ops.append(A.GaussianBlur(blur_limit=(3, blur_max), p=min(max(p_blur, 0.0), 1.0)))

    p_noise = _p("p_noise", 0.0)
    if p_noise > 0:
        std_max = float(params.get("noise_std_max", 20.0))
        # Normalize to [0, 1] range - albumentations expects std as fraction of pixel range
        # If value > 1, assume it's in 0-255 scale and convert
        if std_max > 1.0:
            std_max = std_max / 255.0
        std_max = min(max(std_max, 0.0), 1.0)  # Clamp to valid range
        ops.append(A.GaussNoise(std_range=(0.0, std_max), p=min(max(p_noise, 0.0), 1.0)))

    p_dropout = _p("p_dropout", 0.0)
    if p_dropout > 0:
        holes_max = int(params.get("dropout_holes_max", 4))
        hole_min = float(params.get("dropout_hole_min", 0.05))
        hole_max = float(params.get("dropout_hole_max", 0.15))
        ops.append(
            A.CoarseDropout(
                num_holes_range=(1, max(1, holes_max)),
                hole_height_range=(hole_min, hole_max),
                hole_width_range=(hole_min, hole_max),
                p=min(max(p_dropout, 0.0), 1.0),
            )
        )

    if not ops:
        return None
    return A.Compose(ops)


def resolve_train_augmentation(
    augmentation: str,
    augmentation_params: Mapping[str, Any] | None = None,
) -> Optional[A.Compose]:
    if augmentation in {None, "none"}:
        return None
    if augmentation == "recipe":
        return build_recipe_augmentation(augmentation_params or {})
    return get_augmentation(augmentation)


@register_augmentation("basic")
def aug_basic():
    """Basic augmentations: flip + slight color jitter."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
    ])


@register_augmentation("moderate")
def aug_moderate():
    """Moderate augmentations: flip + color + rotation + blur."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
    ])


@register_augmentation("strong")
def aug_strong():
    """Strong augmentations: includes CoarseDropout, ShiftScale, etc."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, 
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussNoise(std_range=(10, 30), p=1.0),
        ], p=0.4),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15), 
                        hole_width_range=(0.05, 0.15), p=0.3),
    ])




# =============================================================================
# DATASET CLASS (OPTIMIZED)
# =============================================================================

class BirdDataset(Dataset):
    """PyTorch Dataset for bird classification with EfficientNet-B4.

    Preprocessing modes:
    - 'native': Optional pad-to-square → EfficientNet weights.transforms()
    - 'bbox_crop': Bbox crop (+padding) → Optional pad-to-square → weights.transforms()

    Caching (optional):
    - Caches ONLY the deterministic base image after (optional) bbox-crop + (optional) pad-to-square.
    - Augmentations are NOT cached (they stay random each epoch).
    
    OPTIMIZATION: Uses apply_bbox_crop_optimized() to avoid redundant DeepLake
    image loading. The image and box are loaded once per sample.
    """

    def __init__(
        self,
        ds,
        indices: np.ndarray,
        labels: np.ndarray,
        weights,
        preprocess_mode: str = "native",
        bbox_padding_ratio: float = 0.15,
        pad_to_square: bool = True,
        augmentation: Optional[A.Compose] = None,
        return_index: bool = False,
        cache_dir: Path | None = None,
        cache_version: str = "v1_uint8_rgb_pad_bbox",
    ):
        self.ds = ds
        self.indices = np.asarray(indices, dtype=int)
        self.labels = np.asarray(labels, dtype=int)

        self.preprocess_mode = preprocess_mode
        self.bbox_padding_ratio = float(bbox_padding_ratio)
        self.pad_to_square = bool(pad_to_square)
        self.augmentation = augmentation
        self.return_index = bool(return_index)

        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_version = str(cache_version)

        if preprocess_mode not in {"native", "bbox_crop"}:
            raise ValueError("preprocess_mode must be 'native' or 'bbox_crop'")

        self.model_transform = weights.transforms()

    def _cache_path(self, idx: int) -> Path | None:
        if self.cache_dir is None:
            return None
        subdir = self.cache_dir / self.cache_version / self.preprocess_mode
        subdir.mkdir(parents=True, exist_ok=True)
        pad_flag = 1 if self.pad_to_square else 0
        pad_pct = int(round(self.bbox_padding_ratio * 1000))
        return subdir / f"{idx}_pad{pad_flag}_p{pad_pct}.png"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, int]:
        idx = int(self.indices[i])

        cache_path = self._cache_path(idx)
        img = None
        
        # Try to load from cache (with corruption handling)
        if cache_path is not None and cache_path.exists():
            try:
                img = _ensure_uint8_rgb(np.array(Image.open(cache_path).convert("RGB")))
            except (OSError, IOError) as e:
                # Corrupted/truncated cache file - delete and regenerate
                try:
                    cache_path.unlink()
                except Exception:
                    pass
                img = None  # Will be regenerated below
        
        # Generate image if not loaded from cache
        if img is None:
            # OPTIMIZED: Load sample once, get both image and box
            sample = self.ds[idx]
            img = _ensure_uint8_rgb(sample["images"].numpy())

            if self.preprocess_mode == "bbox_crop":
                # OPTIMIZED: Use pre-loaded box array instead of re-fetching from dataset
                box = sample["boxes"].numpy()
                img = apply_bbox_crop_optimized(img, box, padding_ratio=self.bbox_padding_ratio)
                img = _ensure_uint8_rgb(img)

            if self.pad_to_square:
                img = _pad_to_square(img)

            # Save to cache (with error handling)
            if cache_path is not None:
                try:
                    Image.fromarray(img).save(cache_path, format="PNG")
                except Exception:
                    pass  # Silently fail cache write

        if self.augmentation is not None:
            img = self.augmentation(image=img)["image"]
            img = _ensure_uint8_rgb(img)

        x = self.model_transform(Image.fromarray(img))
        y = int(self.labels[i])
        if self.return_index:
            return x, y, idx
        return x, y


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_efficientnet_b4(num_classes: int) -> Tuple[nn.Module, object]:
    """
    Build EfficientNet-B4 with a new classification head.
    
    Architecture:
    - Backbone: EfficientNet-B4 pretrained on ImageNet
    - Classifier: Dropout(0.4) → Linear(1792 → num_classes)
    
    Returns:
        (model, weights) tuple where weights contains the preprocessing transforms
    """
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    
    # Replace classifier (original: Linear(1792 → 1000))
    in_features = model.classifier[-1].in_features  # 1792
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model, weights


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all layers except the classifier head.
    
    Used in Stage 1 of training to train only the randomly initialized head
    while keeping pretrained backbone weights fixed.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all layers for fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True


def set_batchnorm_eval(module: nn.Module) -> None:
    """Set BatchNorm layers to eval mode (keeps running stats frozen)."""
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


# =============================================================================
# OPTIONAL: torch.compile() for PyTorch 2.0+ (10-30% speedup)
# =============================================================================

def maybe_compile_model(model: nn.Module, enable: bool = True) -> nn.Module:
    """
    Compile model with torch.compile for 10-30% speedup (PyTorch 2.0+).
    Falls back gracefully on older versions or unsupported platforms.
    """
    if not enable:
        return model
    
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            compiled = torch.compile(model, mode='reduce-overhead')
            print("torch.compile enabled (10-30% speedup)")
            return compiled
        except Exception as e:
            print(f"torch.compile failed, using eager mode: {e}")
            return model
    return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a weighted sampler for class imbalance.
    
    Classes with fewer samples get higher sampling probability,
    ensuring each class is seen roughly equally often per epoch.
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels)
    counts[counts == 0] = 1  # Avoid division by zero
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler: Optional[WeightedRandomSampler] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with appropriate settings.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if sampler is provided)
        sampler: Optional weighted sampler
        num_workers: Number of worker processes. Use 0 for DeepLake without caching,
                     or 4 when using cached PNG files for ~3x speedup.
    """
    import sys
    import platform
    
    # Use spawn context on Linux (Colab) to avoid fork-related multiprocessing errors
    # The "can only test a child process" assertion error occurs with fork + notebooks
    mp_context = None
    if num_workers > 0 and platform.system() == "Linux":
        import multiprocessing
        mp_context = multiprocessing.get_context("spawn")
    
    # Note: persistent_workers can cause "Bad file descriptor" errors on cleanup,
    # so we disable it. The slight performance hit is worth the stability.
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,  # Disabled to avoid OSError on cleanup
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch for speed
        multiprocessing_context=mp_context,
    )


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute classification metrics."""
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []

    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        
        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs
        
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, n)
    return metrics


@torch.no_grad()
def evaluate_with_preds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    topk: int = 5,
) -> Tuple[Dict[str, float], List[int], List[int], List[int] | None]:
    """Evaluate model and also return (y_true, y_pred) for detailed reports."""
    model.eval()
    total_loss = 0.0
    n = 0
    topk_correct = 0
    y_true, y_pred = [], []
    indices: List[int] | None = []

    for batch in loader:
        x, y = batch[0], batch[1]
        idx = batch[2] if len(batch) >= 3 else None
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs

        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        if indices is not None:
            if idx is None:
                indices = None
            else:
                if torch.is_tensor(idx):
                    indices.extend([int(v) for v in idx.cpu().tolist()])
                else:
                    indices.extend([int(v) for v in idx])

        k = int(min(max(1, topk), logits.shape[1]))
        topk_idx = logits.topk(k=k, dim=1).indices
        topk_correct += int((topk_idx == y[:, None]).any(dim=1).sum().item())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, n)
    metrics["top5_acc"] = float(topk_correct / max(1, n))
    return metrics, y_true, y_pred, indices


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    freeze_bn: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    # Keep BatchNorm in eval mode if requested (recommended for fine-tuning)
    if freeze_bn:
        model.apply(set_batchnorm_eval)
    
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None
    
    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []

    # GPU memory debugging
    if device.type == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"    GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", flush=True)

    # NOTE: First batch may be slow due to DeepLake cloud download + cache building
    print("    Loading first batch...", flush=True)
    pbar = tqdm(loader, desc="Train", leave=False)
    first_batch = True
    for batch in pbar:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
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

        # Debug: Print memory after first batch
        if first_batch and device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"    First batch complete. Peak GPU memory: {peak_mem:.2f}GB", flush=True)
            first_batch = False

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        
        pbar.set_postfix(loss=total_loss / n)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, n)
    return metrics


# =============================================================================
# EVALUATION UTILITIES: Training Curves, Confusion Matrix, Classification Report
# =============================================================================

def plot_evaluation_suite(
    history_df: pd.DataFrame,
    y_true: List[int],
    y_pred: List[int],
    class_ids: List[int],
    run_name: str = "",
    *,
    ds=None,
    sample_indices: List[int] | None = None,
    id_to_name: Dict[int, str] | None = None,
    train_label_index: Dict[int, np.ndarray] | None = None,
    top_n_cm_classes: int = 20,
    top_n_pairs: int = 5,
    images_per_pair: int = 3,
    figsize: Tuple[int, int] = (18, 10),
) -> None:
    """
    Comprehensive evaluation visualization after training.
    
    Displays:
    1. Training curves (loss, accuracy, F1)
    2. Confusion matrix (top confused classes)
    3. Classification report summary (worst performing classes)
    
    Args:
        history_df: Training history DataFrame with columns:
                    [stage, epoch, global_epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1]
        y_true: True labels from evaluation
        y_pred: Predicted labels from evaluation
        class_ids: List of class IDs (maps index back to original class ID)
        run_name: Experiment name for title
        id_to_name: Optional mapping from class_id to species name
        train_label_index: Optional label index for finding reference images of predicted class
        top_n_cm_classes: Number of classes to show in confusion matrix detail
        top_n_pairs: Number of confused pairs to show
        images_per_pair: Number of example images per confused pair
        figsize: Figure size
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_ids))),
        output_dict=True,
        zero_division=0,
    )

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(report["macro avg"]["precision"])
    macro_rec = float(report["macro avg"]["recall"])
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])

    best_epoch = int(history_df.loc[history_df["val_f1"].idxmax(), "global_epoch"])
    best_f1 = float(history_df["val_f1"].max())

    summary_df = pd.DataFrame(
        [
            ("Accuracy", acc),
            ("Macro Precision", macro_prec),
            ("Macro Recall", macro_rec),
            ("Macro F1", macro_f1),
            ("Weighted F1", weighted_f1),
            ("Samples", int(len(y_true))),
            ("Classes", int(len(class_ids))),
            (
                "Classes with F1=0",
                int(sum(1 for i in range(len(class_ids)) if report.get(str(i), {}).get("f1-score", 1.0) == 0)),
            ),
            ("Best Val F1", best_f1),
            ("Best Val F1 Epoch", best_epoch),
        ],
        columns=["Metric", "Value"],
    )

    print("\nEvaluation Summary")
    with pd.option_context("display.max_rows", 200, "display.max_colwidth", 120):
        print(summary_df.to_string(index=False))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], hspace=0.35, wspace=0.3)

    epochs = history_df["global_epoch"]
    head_epochs = (
        history_df[history_df["stage"] == "head"]["global_epoch"].max()
        if "head" in history_df["stage"].values
        else 0
    )

    train_color = "#2ecc71"
    val_color = "#e74c3c"
    val_prec_color = "#3498db"
    val_rec_color = "#f39c12"

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history_df["train_loss"], label="Train", color=train_color, marker="o", markersize=4, linewidth=2)
    ax1.plot(epochs, history_df["val_loss"], label="Val", color=val_color, marker="s", markersize=4, linewidth=2)
    if head_epochs > 0:
        ax1.axvline(x=head_epochs + 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history_df["train_acc"], label="Train", color=train_color, marker="o", markersize=4, linewidth=2)
    ax2.plot(epochs, history_df["val_acc"], label="Val", color=val_color, marker="s", markersize=4, linewidth=2)
    if head_epochs > 0:
        ax2.axvline(x=head_epochs + 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history_df["train_f1"], label="Train F1", color=train_color, marker="o", markersize=4, linewidth=2)
    ax3.plot(epochs, history_df["val_f1"], label="Val F1", color=val_color, marker="s", markersize=4, linewidth=2)

    if "train_precision" in history_df.columns:
        ax3.plot(
            epochs,
            history_df["train_precision"],
            label="Train Precision",
            color=train_color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
        )
    if "train_recall" in history_df.columns:
        ax3.plot(
            epochs,
            history_df["train_recall"],
            label="Train Recall",
            color=train_color,
            linestyle=":",
            linewidth=1.8,
            alpha=0.8,
        )
    if "val_precision" in history_df.columns:
        ax3.plot(
            epochs,
            history_df["val_precision"],
            label="Val Precision",
            color=val_prec_color,
            linestyle="--",
            linewidth=2,
        )
    if "val_recall" in history_df.columns:
        ax3.plot(
            epochs,
            history_df["val_recall"],
            label="Val Recall",
            color=val_rec_color,
            linestyle="--",
            linewidth=2,
        )

    if head_epochs > 0:
        ax3.axvline(x=head_epochs + 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Macro Score")
    ax3.set_title("Macro F1 / Precision / Recall", fontweight="bold")
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    ax3.axvline(x=best_epoch, color="#9b59b6", linestyle=":", alpha=0.8, linewidth=2)
    ax3.scatter([best_epoch], [best_f1], color="#9b59b6", s=80, zorder=5, marker="o")

    ax6 = fig.add_subplot(gs[1, :])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_ids))))
    confused_pairs = []
    for i in range(len(class_ids)):
        for j in range(len(class_ids)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((i, j, int(cm[i, j])))
    confused_pairs_sorted = sorted(confused_pairs, key=lambda x: -x[2])

    top_pairs = confused_pairs_sorted[: max(0, int(top_n_pairs))]

    confused_class_idxs = set()
    for true_i, pred_i, _ in confused_pairs_sorted[: max(0, int(top_n_cm_classes))]:
        confused_class_idxs.add(true_i)
        confused_class_idxs.add(pred_i)

    if not confused_class_idxs:
        # No confusions (or only correct predictions). Show a small subset so the plot still renders.
        confused_class_idxs = set(range(min(int(top_n_cm_classes), len(class_ids))))

    confused_class_idxs = sorted(confused_class_idxs)[: max(2, int(min(top_n_cm_classes, len(class_ids))))]

    cm_subset = cm[np.ix_(confused_class_idxs, confused_class_idxs)]
    tick_labels = [str(class_ids[i]) for i in confused_class_idxs]

    sns.heatmap(
        cm_subset,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax6,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"label": "Count"},
    )
    ax6.set_xlabel("Predicted Class ID", fontsize=10)
    ax6.set_ylabel("True Class ID", fontsize=10)
    ax6.set_title(f"Confusion Matrix for Top {len(confused_class_idxs)} Confused Classes", fontweight="bold")
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax6.set_yticklabels(ax6.get_yticklabels(), rotation=0, fontsize=8)

    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

    if top_pairs:
        print("\nMost Confused Pairs (True -> Predicted)")
        for true_i, pred_i, count in top_pairs:
            true_id = class_ids[true_i]
            pred_id = class_ids[pred_i]
            true_name = id_to_name.get(true_id, f"ID_{true_id}") if id_to_name else f"ID_{true_id}"
            pred_name = id_to_name.get(pred_id, f"ID_{pred_id}") if id_to_name else f"ID_{pred_id}"
            print(f"  {true_name} -> {pred_name}: {count} misclassifications")

    if ds is not None and sample_indices is not None and len(sample_indices) == len(y_true) and top_pairs:
        _plot_confused_pair_examples(
            ds=ds,
            sample_indices=sample_indices,
            y_true=y_true,
            y_pred=y_pred,
            class_ids=class_ids,
            pairs=top_pairs,
            images_per_pair=images_per_pair,
            id_to_name=id_to_name,
            train_label_index=train_label_index,
        )


def _plot_confused_pair_examples(
    *,
    ds,
    sample_indices: List[int],
    y_true: List[int],
    y_pred: List[int],
    class_ids: List[int],
    pairs: List[Tuple[int, int, int]],
    images_per_pair: int = 3,
    id_to_name: Dict[int, str] | None = None,
    train_label_index: Dict[int, np.ndarray] | None = None,
    figsize: Tuple[int, int] | None = None,
) -> None:
    """
    Plot examples of confused pairs with optional reference images of the predicted class.
    
    Each row shows:
    - Misclassified examples (actual class, predicted as wrong class)
    - A reference image of the predicted (wrong) class for comparison
    
    Args:
        ds: DeepLake dataset
        sample_indices: Indices into ds for each prediction
        y_true: True labels (class indices, not IDs)
        y_pred: Predicted labels (class indices, not IDs)
        class_ids: Maps class index to class ID
        pairs: List of (true_idx, pred_idx, count) tuples
        images_per_pair: Number of misclassified examples per pair
        id_to_name: Optional mapping from class_id to species name
        train_label_index: Optional label index to find reference images of predicted class
        figsize: Optional figure size
    """
    images_per_pair = max(1, int(images_per_pair))
    n_pairs = len(pairs)
    if n_pairs <= 0:
        return

    # Add 1 column for reference image if train_label_index is available
    show_reference = train_label_index is not None
    n_cols = images_per_pair + (1 if show_reference else 0)
    
    if figsize is None:
        figsize = (n_cols * 3.2, max(3.0, n_pairs * 2.8))

    fig, axes = plt.subplots(n_pairs, n_cols, figsize=figsize)
    if n_pairs == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row, (true_i, pred_i, count) in enumerate(pairs):
        matches = [k for k, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt == true_i and yp == pred_i]
        chosen = matches[:images_per_pair]
        while len(chosen) < images_per_pair:
            chosen.append(None)

        true_id = class_ids[true_i]
        pred_id = class_ids[pred_i]
        
        # Get species names
        true_name = id_to_name.get(true_id, f"ID_{true_id}") if id_to_name else f"ID_{true_id}"
        pred_name = id_to_name.get(pred_id, f"ID_{pred_id}") if id_to_name else f"ID_{pred_id}"
        
        # Truncate long names for display
        true_name_short = true_name[:25] + "..." if len(true_name) > 28 else true_name
        pred_name_short = pred_name[:25] + "..." if len(pred_name) > 28 else pred_name

        # Plot misclassified examples
        for col, k in enumerate(chosen):
            ax = axes[row, col]
            ax.axis("off")
            if k is None:
                continue
            ds_idx = int(sample_indices[k])
            try:
                sample = ds[ds_idx]
                img = _ensure_uint8_rgb(sample["images"].numpy())
            except Exception:
                continue
            ax.imshow(img)
            if col == 0:
                ax.set_title(f"Actual: {true_name_short}\n→ Pred: {pred_name_short} (n={count})", 
                           fontsize=8, color="red")
            else:
                ax.set_title("Misclassified", fontsize=8, color="red")
        
        # Plot reference image of predicted class (last column)
        if show_reference:
            ref_ax = axes[row, -1]
            ref_ax.axis("off")
            
            # Find a reference image of the predicted class
            if pred_id in train_label_index and len(train_label_index[pred_id]) > 0:
                ref_idx = int(train_label_index[pred_id][0])
                try:
                    ref_sample = ds[ref_idx]
                    ref_img = _ensure_uint8_rgb(ref_sample["images"].numpy())
                    ref_ax.imshow(ref_img)
                    ref_ax.set_title(f"Reference:\n{pred_name_short}", fontsize=8, color="green")
                    # Add a green border to distinguish reference
                    for spine in ref_ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color("green")
                        spine.set_linewidth(3)
                except Exception:
                    ref_ax.text(0.5, 0.5, "Reference\nN/A", ha="center", va="center", fontsize=9)

    fig.suptitle("Confused Pair Examples (with Reference Images)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()



# =============================================================================
# VISUALIZE PREPROCESSING (VERIFY BEFORE TRAINING)
# =============================================================================

def visualize_preprocessing_comparison(
    ds,
    indices: List[int],
    bbox_padding_ratio: float = 0.15,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Visualize the two preprocessing modes side-by-side.
    
    Shows for each sample:
    1. Original image
    2. Native preprocessing (pad-to-square → EfficientNet transforms)
    3. Bbox crop preprocessing (crop → pad-to-square → EfficientNet transforms)
    4. Original with bounding box overlay
    
    IMPORTANT: Run this BEFORE training to verify preprocessing looks correct!
    """
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    
    # Get normalization params for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    def tensor_to_display(t: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor back to displayable image."""
        t_cpu = t.detach().cpu().float()
        # Denormalize
        t_cpu = t_cpu * std + mean
        t_cpu = t_cpu.clamp(0, 1)
        return t_cpu.permute(1, 2, 0).numpy()
    
    n_samples = len(indices)
    fig, axes = plt.subplots(n_samples, 4, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    col_titles = [
        "Original",
        "Native (pad→resize→crop→norm)",
        "Bbox crop (crop→pad→resize→norm)", 
        "Bounding Box Overlay"
    ]
    
    for row, idx in enumerate(indices):
        # Load sample once (OPTIMIZED: avoid multiple DeepLake accesses)
        sample = ds[idx]
        img = _ensure_uint8_rgb(sample["images"].numpy())
        h, w = img.shape[:2]
        box = sample["boxes"].numpy()
        bbox = resolve_bbox_from_box_array(box, h, w)
        class_id = int(sample["labels"].numpy().item())
        
        # Column 0: Original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original ({w}×{h})", fontsize=9)
        axes[row, 0].axis("off")
        
        # Column 1: Native preprocessing
        img_padded = _pad_to_square(img)
        tensor_native = preprocess(Image.fromarray(img_padded))
        native_display = tensor_to_display(tensor_native)
        axes[row, 1].imshow(native_display)
        axes[row, 1].set_title(f"Native (380×380)", fontsize=9)
        axes[row, 1].axis("off")
        
        # Column 2: Bbox crop preprocessing (OPTIMIZED: use pre-loaded box)
        img_bbox = apply_bbox_crop_optimized(img, box, padding_ratio=bbox_padding_ratio)
        img_bbox = _ensure_uint8_rgb(img_bbox)
        img_bbox_padded = _pad_to_square(img_bbox)
        tensor_bbox = preprocess(Image.fromarray(img_bbox_padded))
        bbox_display = tensor_to_display(tensor_bbox)
        axes[row, 2].imshow(bbox_display)
        crop_h, crop_w = img_bbox.shape[:2]
        axes[row, 2].set_title(f"Bbox crop ({crop_w}×{crop_h}→380×380)", fontsize=9)
        axes[row, 2].axis("off")
        
        # Column 3: Original with bbox overlay
        axes[row, 3].imshow(img)
        if bbox is not None:
            x1, y1, x2, y2 = map(float, bbox)
            box_w, box_h = x2 - x1, y2 - y1
            coverage = box_w * box_h / (h * w) * 100
            
            # Draw bbox
            rect = plt.Rectangle((x1, y1), box_w, box_h,
                                  fill=False, edgecolor="lime", linewidth=2)
            axes[row, 3].add_patch(rect)
            
            # Draw padded region
            pad_x = int(box_w * bbox_padding_ratio)
            pad_y = int(box_h * bbox_padding_ratio)
            padded_rect = plt.Rectangle(
                (max(0, x1-pad_x), max(0, y1-pad_y)),
                min(w, x2+pad_x) - max(0, x1-pad_x),
                min(h, y2+pad_y) - max(0, y1-pad_y),
                fill=False, edgecolor="yellow", linewidth=1, linestyle="--"
            )
            axes[row, 3].add_patch(padded_rect)
            axes[row, 3].set_title(f"Bbox: {coverage:.1f}% of image", fontsize=9)
        else:
            axes[row, 3].set_title("No bbox available", fontsize=9)
        axes[row, 3].axis("off")
        
        # Row label
        axes[row, 0].set_ylabel(f"Class {class_id}\nIdx {idx}", 
                                 fontsize=9, rotation=0, ha="right", va="center", labelpad=40)
    
    # Column titles
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")
    
    plt.suptitle("Preprocessing Comparison: Native vs Bbox Crop\n"
                 "(Green = bbox, Yellow dashed = crop region with padding)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainConfig:
    """
    All hyperparameters for a training experiment.
    
    Attributes:
        run_name: Unique name for this experiment (used for saving)
        preprocess_mode: 'native' or 'bbox_crop'
        augmentation: Name from AUGMENTATION_REGISTRY, 'none', or 'recipe'
        augmentation_params: Parameters for 'recipe' augmentation
        bbox_padding_ratio: Padding around bbox as fraction of bbox size
        pad_to_square: Whether to pad images to square before model transforms
        
        batch_size: Training batch size
        head_epochs: Number of epochs to train head only (Stage 1)
        finetune_epochs: Number of epochs to fine-tune all layers (Stage 2)
        
        lr_head: Learning rate for head during Stage 1
        lr_backbone: Learning rate for backbone during Stage 2
        lr_head_finetune: Learning rate for head during Stage 2
        weight_decay: AdamW weight decay
        label_smoothing: Label smoothing for CrossEntropyLoss
        
        use_weighted_sampler: Whether to use weighted sampling for class imbalance
        use_amp: Whether to use automatic mixed precision (faster on GPU)
        use_torch_compile: Whether to use torch.compile for speedup (PyTorch 2.0+)
        grad_clip_norm: Gradient clipping max norm
        
        freeze_bn_head: Keep BatchNorm in eval mode during Stage 1
        freeze_bn_finetune: Keep BatchNorm in eval mode during Stage 2

        resume_head_ckpt: Optional path to a previously saved best head checkpoint (.pt).
          If set, Stage 1 is skipped and training starts from that checkpoint.
        
        early_stop_patience: Stop if val F1 doesn't improve for N epochs (0 = disabled)
        
        seed: Random seed
    """
    # Experiment identification
    run_name: str = "effnetb4_baseline_v1"
    
    # Preprocessing
    preprocess_mode: str = "native"  # 'native' or 'bbox_crop'
    augmentation: str = "none"  # 'none', registry name, or 'recipe'
    augmentation_params: Dict[str, Any] | None = None
    bbox_padding_ratio: float = 0.15
    pad_to_square: bool = True

    # Caching (optional): persistent cache dir (e.g. Google Drive)
    cache_dir: str | None = None  # e.g. str(CACHE_DIR)
    cache_version: str = "v1_uint8_rgb_pad_bbox"
    
    # Training schedule
    batch_size: int = 16
    head_epochs: int = 3
    finetune_epochs: int = 10
    
    # Learning rates
    lr_head: float = 3e-3
    lr_backbone: float = 3e-5
    lr_head_finetune: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    
    # Training options
    use_weighted_sampler: bool = True
    use_amp: bool = True
    use_torch_compile: bool = True  # PyTorch 2.0+ speedup (10-30%)
    grad_clip_norm: float = 1.0
    
    # BatchNorm handling
    freeze_bn_head: bool = True
    freeze_bn_finetune: bool = True

    # Resume (optional)
    resume_head_ckpt: str | None = None
    
    # Early stopping (0 = disabled)
    early_stop_patience: int = 3
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.preprocess_mode not in {"native", "bbox_crop"}:
            raise ValueError(f"preprocess_mode must be 'native' or 'bbox_crop', got '{self.preprocess_mode}'")
        if self.augmentation == "recipe":
            if not isinstance(self.augmentation_params, dict) or not self.augmentation_params:
                raise ValueError("augmentation_params must be a non-empty dict when augmentation='recipe'")
            return
        if self.augmentation not in AUGMENTATION_REGISTRY and self.augmentation != "none":
            raise ValueError(
                f"augmentation must be one of {list(AUGMENTATION_REGISTRY.keys())}, 'recipe', or 'none'"
            )




# =============================================================================
# TWO-STAGE TRAINING FUNCTION
# =============================================================================

def train_two_stage(
    cfg: TrainConfig,
    ds_train,
    train_label_index: Dict[int, np.ndarray],
    ds_holdout=None,
    holdout_label_index: Optional[Dict[int, np.ndarray]] = None,
    *,
    splits=None,
    evaluate_test: bool = True,
    evaluate_holdout: bool = False,
    on_epoch_end: Callable[[str, Dict[str, float], int], None] | None = None,
    device: torch.device = DEVICE,
    output_dir: Path = RUNS_DIR,
) -> Tuple[nn.Module, pd.DataFrame, Dict[str, Any], Path]:
    """
    Train EfficientNet-B4 in two stages: head-only, then fine-tune.
    
    Args:
        cfg: Training configuration
        ds_train: Training dataset (will be split into train/val/test)
        train_label_index: Label index for ds_train
        ds_holdout: Optional holdout dataset (ONLY used when evaluate_holdout=True)
        holdout_label_index: Optional label index for ds_holdout
        splits: Optional precomputed train/val/test splits for ds_train
        evaluate_test: Whether to evaluate the internal 15% test split at the end
        evaluate_holdout: Whether to evaluate ds_holdout at the end (RUN ONCE at the very end)
        device: Training device
        output_dir: Directory to save checkpoints and logs
    
    Returns:
        model: Trained model (best checkpoint)
        history: DataFrame with training history
        summary: Dict with final evaluation metrics
        run_dir: Path to run directory
    """
    seed_everything(cfg.seed)
    
    # Setup output directory
    run_dir = output_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    
    # Class mapping
    class_ids = sorted(train_label_index.keys())
    class_id_to_idx = {cid: i for i, cid in enumerate(class_ids)}
    num_classes = len(class_ids)
    
    # Create / reuse data splits (keep experiments comparable)
    if splits is None:
        splits = split_label_index_stratified(
            train_label_index,
            class_id_to_idx=class_id_to_idx,
            val_frac=0.15,
            test_frac=0.15,
            seed=cfg.seed,
        )
    
    print(f"Splits: train={len(splits.train_idx)}, val={len(splits.val_idx)}, test={len(splits.test_idx)}")
    
    # Build model
    model, weights = build_efficientnet_b4(num_classes)
    model = model.to(device)

    # Apply torch.compile for speedup (PyTorch 2.0+)
    if cfg.use_torch_compile:
        model = maybe_compile_model(model, enable=True)

    resumed_from_head = False
    if cfg.resume_head_ckpt:
        resume_path = Path(cfg.resume_head_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_head_ckpt not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise ValueError(f"Invalid head checkpoint (missing 'model_state'): {resume_path}")
        model.load_state_dict(ckpt["model_state"])
        resumed_from_head = True

    # Get augmentation
    train_aug = resolve_train_augmentation(cfg.augmentation, cfg.augmentation_params)

    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    
    # Create datasets
    train_ds = BirdDataset(
        ds_train, splits.train_idx, splits.train_y,
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        pad_to_square=cfg.pad_to_square,
        augmentation=train_aug,
        cache_dir=cache_dir,
        cache_version=cfg.cache_version,
    )
    val_ds = BirdDataset(
        ds_train, splits.val_idx, splits.val_y,
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        pad_to_square=cfg.pad_to_square,
        augmentation=None,  # No augmentation for validation
        cache_dir=cache_dir,
        cache_version=cfg.cache_version,
    )
    test_ds = BirdDataset(
        ds_train, splits.test_idx, splits.test_y,
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        pad_to_square=cfg.pad_to_square,
        augmentation=None,
        return_index=True,
        cache_dir=cache_dir,
        cache_version=cfg.cache_version,
    )
    
    # Create dataloaders
    # Use num_workers=4 when caching is enabled (cached PNGs don't need DeepLake)
    num_workers = 4 if cache_dir is not None else 0
    sampler = make_weighted_sampler(splits.train_y) if cfg.use_weighted_sampler else None
    train_loader = make_dataloader(train_ds, cfg.batch_size, shuffle=True, sampler=sampler, num_workers=num_workers)
    val_loader = make_dataloader(val_ds, cfg.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_dataloader(test_ds, cfg.batch_size, shuffle=False, num_workers=num_workers)
    
    holdout_loader = None
    if evaluate_holdout:
        if ds_holdout is None or holdout_label_index is None:
            raise ValueError("evaluate_holdout=True requires ds_holdout and holdout_label_index")
        holdout_indices, holdout_labels = [], []
        for class_id, idxs in holdout_label_index.items():
            if class_id not in class_id_to_idx:
                continue
            holdout_indices.extend(idxs.tolist())
            holdout_labels.extend([class_id_to_idx[class_id]] * len(idxs))
        holdout_ds = BirdDataset(
            ds_holdout,
            np.array(holdout_indices),
            np.array(holdout_labels),
            weights=weights,
            preprocess_mode=cfg.preprocess_mode,
            bbox_padding_ratio=cfg.bbox_padding_ratio,
            pad_to_square=cfg.pad_to_square,
            augmentation=None,
            return_index=True,
            cache_dir=cache_dir,
            cache_version=cfg.cache_version,
        )
        holdout_loader = make_dataloader(holdout_ds, cfg.batch_size, shuffle=False, num_workers=num_workers)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    
    history = []
    
    # ==========================================================================
    # NESTED FUNCTION: run_stage
    # ==========================================================================
    def run_stage(
        stage_name: str,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        freeze_bn: bool,
        epoch_offset: int = 0,
    ) -> Path:
        """Run one training stage and return path to best checkpoint."""
        best_f1 = -1.0
        best_path = run_dir / f"best_{stage_name}.pt"
        no_improve = 0
        
        # Epoch-level progress bar with time estimates
        epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Stage: {stage_name}", unit="epoch")
        
        for ep in epoch_pbar:
            t0 = time.time()
            
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, device, criterion, optimizer,
                use_amp=cfg.use_amp,
                grad_clip_norm=cfg.grad_clip_norm,
                freeze_bn=freeze_bn,
            )
            
            # Validate
            val_metrics = evaluate(model, val_loader, device, criterion)
            
            # Log
            elapsed = time.time() - t0
            row = {
                "stage": stage_name,
                "epoch": ep,
                "global_epoch": epoch_offset + ep,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "seconds": elapsed,
            }
            history.append(row)
            
            # Update progress bar with metrics
            epoch_pbar.set_postfix({
                "val_f1": f"{val_metrics['f1']:.3f}",
                "val_loss": f"{val_metrics['loss']:.3f}",
                "best_f1": f"{max(best_f1, val_metrics['f1']):.3f}",
            })

            if on_epoch_end is not None:
                on_epoch_end(stage_name, val_metrics, epoch_offset + ep)
            
            # Checkpoint
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "class_ids": class_ids,
                    "class_id_to_idx": class_id_to_idx,
                    "epoch": epoch_offset + ep,
                    "val_f1": best_f1,
                    "config": asdict(cfg),
                }, best_path)
            else:
                no_improve += 1
            
            # Early stopping
            if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
                print(f"  Early stopping triggered (no improvement for {cfg.early_stop_patience} epochs)")
                break
        
        return best_path
    
    # ==========================================================================
    # STAGE 1: Head Training (Frozen Backbone)
    # ==========================================================================
    best_head_path: Path | None = None
    finetune_epoch_offset = 0
    if resumed_from_head:
        print(f"\n{'='*60}")
        print("STAGE 1: Skipped (Resuming from best head checkpoint)")
        print(f"{'='*60}")
        print(f"Resume: {cfg.resume_head_ckpt}")
        best_head_path = Path(cfg.resume_head_ckpt)
        finetune_epoch_offset = 0
    elif cfg.head_epochs > 0:
        print(f"\n{'='*60}")
        print("STAGE 1: Training Classification Head (Backbone Frozen)")
        print(f"{'='*60}")

        freeze_backbone(model)
        optimizer_head = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr_head,
            weight_decay=cfg.weight_decay,
        )

        best_head_path = run_stage(
            "head",
            cfg.head_epochs,
            optimizer_head,
            freeze_bn=cfg.freeze_bn_head,
            epoch_offset=0,
        )

        # Load best head checkpoint
        model.load_state_dict(torch.load(best_head_path, map_location=device)["model_state"])
        finetune_epoch_offset = cfg.head_epochs
    else:
        print(f"\n{'='*60}")
        print("STAGE 1: Skipped (head_epochs=0)")
        print(f"{'='*60}")
        finetune_epoch_offset = 0
    
    # ==========================================================================
    # STAGE 2: Fine-tuning (All Layers) — only if finetune_epochs > 0
    # ==========================================================================
    if cfg.finetune_epochs > 0:
        print(f"\n{'='*60}")
        print("STAGE 2: Fine-tuning All Layers")
        print(f"{'='*60}")
        
        unfreeze_all(model)
        
        # Different learning rates for backbone vs head
        head_params = list(model.classifier.parameters())
        head_param_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
        
        optimizer_finetune = torch.optim.AdamW([
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head_finetune},
        ], weight_decay=cfg.weight_decay)
        
        best_finetune_path = run_stage(
            "finetune", cfg.finetune_epochs, optimizer_finetune,
            freeze_bn=cfg.freeze_bn_finetune, epoch_offset=finetune_epoch_offset
        )
        
        # Load best fine-tuned checkpoint
        model.load_state_dict(torch.load(best_finetune_path, map_location=device)["model_state"])
    else:
        print(f"\n{'='*60}")
        print("STAGE 2: Skipped (finetune_epochs=0)")
        print(f"{'='*60}")
    
    # ==========================================================================
    # FINAL EVALUATION
    # ==========================================================================
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    test_metrics = None
    test_y_true, test_y_pred = None, None
    if evaluate_test:
        test_metrics, test_y_true, test_y_pred, test_indices = evaluate_with_preds(model, test_loader, device, criterion)
        (run_dir / "test_classification_report.txt").write_text(
            classification_report(
                test_y_true,
                test_y_pred,
                labels=list(range(num_classes)),
                target_names=[str(cid) for cid in class_ids],
                digits=3,
                zero_division=0,
            )
        )
        rep = classification_report(
            test_y_true, test_y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0
        )
        test_metrics["weighted_precision"] = float(rep["weighted avg"]["precision"])
        test_metrics["weighted_recall"] = float(rep["weighted avg"]["recall"])
        test_metrics["weighted_f1"] = float(rep["weighted avg"]["f1-score"])
        print("\n" + "=" * 70)
        print("TEST SET (15% from train split)")
        print("=" * 70)
        print(f"  Accuracy:         {test_metrics['acc']:.4f}")
        print(f"  Top-5 Accuracy:   {test_metrics['top5_acc']:.4f}")
        print(f"  Macro Precision:  {test_metrics['precision']:.4f}")
        print(f"  Macro Recall:     {test_metrics['recall']:.4f}")
        print(f"  Macro F1:         {test_metrics['f1']:.4f}")
        print(f"  Weighted F1:      {test_metrics['weighted_f1']:.4f}")
    
    holdout_metrics = None
    holdout_y_true, holdout_y_pred = None, None
    if evaluate_holdout:
        holdout_metrics, holdout_y_true, holdout_y_pred, holdout_indices = evaluate_with_preds(
            model, holdout_loader, device, criterion
        )
        (run_dir / "holdout_classification_report.txt").write_text(
            classification_report(
                holdout_y_true,
                holdout_y_pred,
                labels=list(range(num_classes)),
                target_names=[str(cid) for cid in class_ids],
                digits=3,
                zero_division=0,
            )
        )
        rep = classification_report(
            holdout_y_true, holdout_y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0
        )
        holdout_metrics["weighted_precision"] = float(rep["weighted avg"]["precision"])
        holdout_metrics["weighted_recall"] = float(rep["weighted avg"]["recall"])
        holdout_metrics["weighted_f1"] = float(rep["weighted avg"]["f1-score"])
        print("\n" + "=" * 70)
        print("HOLDOUT SET (NABirds validation split)")
        print("=" * 70)
        print(f"  Accuracy:         {holdout_metrics['acc']:.4f}")
        print(f"  Top-5 Accuracy:   {holdout_metrics['top5_acc']:.4f}")
        print(f"  Macro Precision:  {holdout_metrics['precision']:.4f}")
        print(f"  Macro Recall:     {holdout_metrics['recall']:.4f}")
        print(f"  Macro F1:         {holdout_metrics['f1']:.4f}")
        print(f"  Weighted F1:      {holdout_metrics['weighted_f1']:.4f}")
    
    # Save results
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    
    summary = {
        "test": test_metrics,
        "holdout": holdout_metrics,
        "val_best_f1": float(history_df["val_f1"].max()),
        "class_ids": class_ids,  # Include class_ids for evaluation suite
        "test_y_true": test_y_true,
        "test_y_pred": test_y_pred,
        "test_indices": test_indices if evaluate_test else None,
        "holdout_y_true": holdout_y_true,
        "holdout_y_pred": holdout_y_pred,
        "holdout_indices": holdout_indices if evaluate_holdout else None,
    }
    (run_dir / "summary.json").write_text(json.dumps({
        "test": test_metrics,
        "holdout": holdout_metrics,
        "val_best_f1": float(history_df["val_f1"].max()),
    }, indent=2))
    
    print(f"\nResults saved to: {run_dir}")
    
    return model, history_df, summary, run_dir






def plot_training_history(history_df: pd.DataFrame, run_name: str = "") -> None:
    """
    Plot comprehensive training curves from history DataFrame.
    
    Shows 4 subplots:
    1. Loss (train vs val)
    2. Accuracy (train vs val) 
    3. F1 Score (train vs val)
    4. All validation metrics combined
    
    A vertical dashed line marks the transition from Stage 1 (head) to Stage 2 (finetune).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    epochs = history_df["global_epoch"]
    
    # Find stage boundary for vertical line
    head_epochs = history_df[history_df["stage"] == "head"]["global_epoch"].max()
    
    # Color scheme
    train_color = "#2ecc71"  # Green
    val_color = "#e74c3c"    # Red
    
    # =========================================================================
    # Plot 1: Loss
    # =========================================================================
    ax = axes[0]
    ax.plot(epochs, history_df["train_loss"], label="Train Loss", 
            color=train_color, marker="o", markersize=5, linewidth=2)
    ax.plot(epochs, history_df["val_loss"], label="Val Loss", 
            color=val_color, marker="s", markersize=5, linewidth=2)
    ax.axvline(x=head_epochs + 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Loss", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add stage labels
    ax.text(head_epochs / 2, ax.get_ylim()[1] * 0.95, "Head", ha="center", fontsize=9, alpha=0.7)
    ax.text(head_epochs + (epochs.max() - head_epochs) / 2, ax.get_ylim()[1] * 0.95, 
            "Finetune", ha="center", fontsize=9, alpha=0.7)
    
    # =========================================================================
    # Plot 2: Accuracy
    # =========================================================================
    ax = axes[1]
    ax.plot(epochs, history_df["train_acc"], label="Train Accuracy", 
            color=train_color, marker="o", markersize=5, linewidth=2)
    ax.plot(epochs, history_df["val_acc"], label="Val Accuracy", 
            color=val_color, marker="s", markersize=5, linewidth=2)
    ax.axvline(x=head_epochs + 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Accuracy", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # =========================================================================
    # Plot 3: F1 Score
    # =========================================================================
    ax = axes[2]
    ax.plot(epochs, history_df["train_f1"], label="Train F1", 
            color=train_color, marker="o", markersize=5, linewidth=2)
    ax.plot(epochs, history_df["val_f1"], label="Val F1", 
            color=val_color, marker="s", markersize=5, linewidth=2)
    ax.axvline(x=head_epochs + 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Macro F1 Score", fontsize=10)
    ax.set_title("F1 Score (Macro)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # =========================================================================
    # Plot 4: All Validation Metrics Combined
    # =========================================================================
    ax = axes[3]
    ax.plot(epochs, history_df["val_acc"], label="Val Accuracy", 
            color="#3498db", marker="o", markersize=5, linewidth=2)
    ax.plot(epochs, history_df["val_f1"], label="Val F1", 
            color="#e74c3c", marker="s", markersize=5, linewidth=2)
    ax.axvline(x=head_epochs + 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Validation Metrics Overview", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Highlight best epoch
    best_epoch = history_df.loc[history_df["val_f1"].idxmax(), "global_epoch"]
    best_f1 = history_df["val_f1"].max()
    ax.axvline(x=best_epoch, color="#9b59b6", linestyle=":", alpha=0.8, linewidth=2)
    ax.scatter([best_epoch], [best_f1], color="#9b59b6", s=100, zorder=5, marker="*")
    ax.annotate(f"Best: {best_f1:.3f}", xy=(best_epoch, best_f1), 
                xytext=(best_epoch + 0.5, best_f1 - 0.05),
                fontsize=9, color="#9b59b6")
    
    # Main title
    title = "Training History"
    if run_name:
        title += f" — {run_name}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nTraining Summary:")
    print(f"   Best Val F1: {history_df['val_f1'].max():.4f} (epoch {int(best_epoch)})")
    print(f"   Best Val Acc: {history_df['val_acc'].max():.4f}")
    print(f"   Final Train Loss: {history_df['train_loss'].iloc[-1]:.4f}")
    print(f"   Final Val Loss: {history_df['val_loss'].iloc[-1]:.4f}")





# =============================================================================
# EXPERIMENT RUNNER (simple, repeatable experiments)
# =============================================================================



def make_experiment_runner(
    *,
    ds_train,
    train_label_index: Dict[int, np.ndarray],
    splits,
    ds_holdout=None,
    holdout_label_index: Optional[Dict[int, np.ndarray]] = None,
    output_dir: Path = None,
):
    """Create a tiny runner so you can launch experiments with 1 line.

    Rules enforced by default:
      - Uses the SAME `splits` for every run (fair comparisons).
      - Uses ONLY internal val/test from `ds_train` during experiments.
      - Holdout (`ds_val`) is only used when you explicitly call `final_holdout(...)`.
    
    Args:
        output_dir: Directory for saving experiment results. Defaults to RUNS_DIR.
                    Pass a Google Drive path in Colab for persistence.
    """
    # Use provided output_dir or fall back to default RUNS_DIR
    runs_dir = output_dir if output_dir is not None else RUNS_DIR
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    
    # Build id_to_name mapping from dataset metadata
    id_to_name: Dict[int, str] | None = None
    try:
        class_names = ds_train.labels.info.get("class_names", None)
        if class_names is not None:
            actual_labels_used = set(train_label_index.keys())
            id_to_name = {
                label_id: class_names[label_id]
                for label_id in actual_labels_used
                if label_id < len(class_names)
            }
            print(f"Loaded species names for {len(id_to_name)} classes")
    except Exception as e:
        print(f"Could not load species names: {e}")
        id_to_name = None

    def _cleanup(model: nn.Module) -> None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(
        run_name: str,
        *,
        plot: bool = True,
        evaluate_test: bool = True,
        **cfg_overrides,
    ):
        cfg_kwargs = {**BASELINE_CONFIG, **cfg_overrides}
        cfg = TrainConfig(run_name=run_name, **cfg_kwargs)

        model, history_df, summary, run_dir = train_two_stage(
            cfg=cfg,
            ds_train=ds_train,
            train_label_index=train_label_index,
            splits=splits,
            evaluate_test=evaluate_test,
            evaluate_holdout=False,
            output_dir=runs_dir,
        )

        if plot:
            # Show comprehensive evaluation suite: training curves + confusion matrix + classification report
            if evaluate_test and summary.get("test_y_true") is not None:
                plot_evaluation_suite(
                    history_df=history_df,
                    y_true=summary["test_y_true"],
                    y_pred=summary["test_y_pred"],
                    class_ids=summary["class_ids"],
                    run_name=cfg.run_name,
                    ds=ds_train,
                    sample_indices=summary.get("test_indices"),
                    id_to_name=id_to_name,
                    train_label_index=train_label_index,
                )
            else:
                # Fallback to simple training history plot if no test predictions
                plot_training_history(history_df, run_name=cfg.run_name)

        test = summary.get("test") or {}
        row = {
            "run_name": cfg.run_name,
            "preprocess_mode": cfg.preprocess_mode,
            "augmentation": cfg.augmentation,
            "augmentation_params": cfg.augmentation_params,
            "resume_head_ckpt": cfg.resume_head_ckpt,
            "val_best_f1": summary.get("val_best_f1"),
            "test_f1": test.get("f1"),
            "test_precision": test.get("precision"),
            "test_recall": test.get("recall"),
            "test_acc": test.get("acc"),
            "test_top5_acc": test.get("top5_acc"),
            "test_weighted_f1": test.get("weighted_f1"),
            "test_weighted_precision": test.get("weighted_precision"),
            "test_weighted_recall": test.get("weighted_recall"),
            "run_dir": str(run_dir),
        }
        results.append(row)

        _cleanup(model)
        return row, history_df, summary, run_dir

    def run_many(
        experiments: Dict[str, Dict[str, Any]],
        *,
        plot: bool = False,
        evaluate_test: bool = True,
        **common_overrides,
    ) -> None:
        """Run a batch of experiments.

        Example:
          runner.run_many({
              "lr_1e-3": {"lr_head": 1e-3},
              "lr_3e-3": {"lr_head": 3e-3},
          }, preprocess_mode="bbox_crop")
        """
        for run_name, overrides in experiments.items():
            merged = {**common_overrides, **(overrides or {})}
            run(run_name, plot=plot, evaluate_test=evaluate_test, **merged)

    def df(*, sort_by: str = "val_best_f1", ascending: bool = False) -> pd.DataFrame:
        if not results:
            return pd.DataFrame([])
        return pd.DataFrame(results).sort_values(sort_by, ascending=ascending)

    def reset() -> None:
        results.clear()

    def load_cached(run_name: str, plot: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load a previously completed experiment from disk and add to results.
        
        Returns the result dict if found, None otherwise.
        Use this to restore experiments from previous sessions.
        """
        saved = load_experiment_results(run_name, runs_dir=runs_dir)
        if saved is None:
            return None
        
        config = saved["config"]
        summary = saved["summary"]
        history_df = saved["history_df"]
        test = summary.get("test") or {}
        
        row = {
            "run_name": run_name,
            "preprocess_mode": config.get("preprocess_mode"),
            "augmentation": config.get("augmentation"),
            "augmentation_params": config.get("augmentation_params"),
            "resume_head_ckpt": config.get("resume_head_ckpt"),
            "val_best_f1": summary.get("val_best_f1"),
            "test_f1": test.get("f1"),
            "test_precision": test.get("precision"),
            "test_recall": test.get("recall"),
            "test_acc": test.get("acc"),
            "test_top5_acc": test.get("top5_acc"),
            "test_weighted_f1": test.get("weighted_f1"),
            "test_weighted_precision": test.get("weighted_precision"),
            "test_weighted_recall": test.get("weighted_recall"),
            "run_dir": str(saved["run_dir"]),
        }
        results.append(row)
        
        if plot:
            plot_training_history(history_df, run_name=run_name)
        
        print(f"  Loaded cached experiment: {run_name} (val_f1={summary.get('val_best_f1', 0):.4f})")
        return row

    def run_or_load(
        run_name: str,
        *,
        plot: bool = True,
        evaluate_test: bool = True,
        force_retrain: bool = False,
        **cfg_overrides,
    ):
        """
        Run experiment OR load from cache if already completed.
        
        This is the recommended method for resumable experiments:
        - Checks if experiment already exists on disk
        - If exists: loads results and skips training
        - If not exists: runs training normally
        
        Args:
            run_name: Experiment name
            plot: Whether to show plots
            evaluate_test: Whether to evaluate on test set
            force_retrain: If True, always retrain even if cached
            **cfg_overrides: Training config overrides
        
        Returns:
            Same as run(): (row, history_df, summary, run_dir)
        """
        if not force_retrain and check_experiment_exists(run_name, runs_dir=runs_dir):
            # Load from cache
            row = load_cached(run_name, plot=plot)
            if row is not None:
                saved = load_experiment_results(run_name, runs_dir=runs_dir)
                return row, saved["history_df"], saved["summary"], saved["run_dir"]
        
        # Run training
        return run(run_name, plot=plot, evaluate_test=evaluate_test, **cfg_overrides)

    def load_all_cached(run_names: List[str] = None, plot: bool = False) -> int:
        """
        Load multiple cached experiments at once.
        
        Args:
            run_names: List of experiment names to load. If None, loads all
                       completed experiments found in runs_dir.
            plot: Whether to show plots for each loaded experiment
        
        Returns:
            Number of experiments successfully loaded
        """
        if run_names is None:
            run_names = get_completed_experiments(runs_dir=runs_dir)
        
        loaded = 0
        for name in run_names:
            if load_cached(name, plot=plot) is not None:
                loaded += 1
        
        return loaded

    def final_holdout(
        run_name: str,
        *,
        plot: bool = True,
        **cfg_overrides,
    ):
        """Run a single FINAL evaluation that includes the holdout (ds_val)."""
        if ds_holdout is None:
            raise ValueError("Pass ds_holdout into make_experiment_runner(...)")

        # Lazily load/build the holdout label index here so experiments never touch holdout labels.
        # Use CLEANED holdout index if available (duplicates removed in EDA)
        holdout_index = holdout_label_index
        val_clean_path = Path("label_index_val_clean.npz")
        holdout_index_path = Path("label_index_holdout.npz")
        
        if holdout_index is None:
            if val_clean_path.exists():
                holdout_index = load_label_index(val_clean_path)
                print(f"Using cleaned holdout index from {val_clean_path}")
            elif holdout_index_path.exists():
                holdout_index = load_label_index(holdout_index_path)
                print(f"Using original holdout index from {holdout_index_path}")
            else:
                holdout_index = build_label_index(ds_holdout)
                save_label_index(holdout_index, holdout_index_path)


        cfg_kwargs = {**BASELINE_CONFIG, **cfg_overrides}
        cfg = TrainConfig(run_name=run_name, **cfg_kwargs)

        model, history_df, summary, run_dir = train_two_stage(
            cfg=cfg,
            ds_train=ds_train,
            train_label_index=train_label_index,
            ds_holdout=ds_holdout,
            holdout_label_index=holdout_index,
            splits=splits,
            evaluate_test=True,
            evaluate_holdout=True,
            output_dir=runs_dir,
        )

        if plot:
            # Show comprehensive evaluation suite for holdout if available
            if summary.get("holdout_y_true") is not None:
                print("\n" + "=" * 70)
                print("HOLDOUT EVALUATION SUITE")
                print("=" * 70)
                plot_evaluation_suite(
                    history_df=history_df,
                    y_true=summary["holdout_y_true"],
                    y_pred=summary["holdout_y_pred"],
                    class_ids=summary["class_ids"],
                    run_name=f"{cfg.run_name} (Holdout)",
                    ds=ds_holdout,
                    sample_indices=summary.get("holdout_indices"),
                    id_to_name=id_to_name,
                    train_label_index=holdout_label_index,  # Use holdout index for reference images
                )
            else:
                plot_training_history(history_df, run_name=cfg.run_name)

        _cleanup(model)
        return history_df, summary, run_dir

    return SimpleNamespace(
        run=run,
        run_or_load=run_or_load,
        run_many=run_many,
        df=df,
        reset=reset,
        load_cached=load_cached,
        load_all_cached=load_all_cached,
        final_holdout=final_holdout,
        results=results,
        runs_dir=runs_dir,
    )
