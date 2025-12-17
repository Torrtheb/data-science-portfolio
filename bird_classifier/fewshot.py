import numpy as np
import cv2
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import os
import sys
import random
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms as T

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from lime import lime_image
import deeplake
import cv2

# Device selection utility

def get_device() -> torch.device:
    """Pick the best available device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_optimized_loader(
    ds: Dataset, 
    batch_size: int, 
    shuffle: bool, 
    sampler=None,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader with GPU optimizations:
    - num_workers: Parallel data loading (reduces CPU-GPU bottleneck)
    - pin_memory: Faster CPU→GPU transfer
    - prefetch_factor: Preload batches while GPU computes
    - persistent_workers: Keep workers alive between epochs
    """
    use_workers = num_workers if torch.cuda.is_available() else 0
    
    return DataLoader(
        ds,
        batch_size=max(1, int(batch_size)),
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=use_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if use_workers > 0 else None,
        persistent_workers=use_workers > 0,
    )

def maybe_compile_model(model: torch.nn.Module, enable: bool = True) -> torch.nn.Module:
    """
    Compile model with torch.compile for 10-30% speedup (PyTorch 2.0+).
    Falls back gracefully on older versions.
    """
    if not enable:
        return model
    
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            compiled = torch.compile(model, mode='reduce-overhead')
            print("✓ torch.compile enabled (10-30% speedup)")
            return compiled
        except Exception as e:
            print(f"⚠️ torch.compile failed, using eager mode: {e}")
            return model
    return model



# Bounding box utilities


def resolve_bbox_from_box_array(
    box: np.ndarray, 
    img_h: int, 
    img_w: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert bbox to pixel-space (x1, y1, x2, y2) clipped to image bounds.
    Handles (x,y,w,h), (x1,y1,x2,y2), and normalized corners.
    
    OPTIMIZED: Takes pre-loaded box array and image dimensions to avoid
    redundant image loading.
    Handles:
    - Normalized xyxy (0-1 range)
    - xywh format
    - xyxy format
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


def resolve_bbox_xywh_or_xyxy(ds, idx: int):
    """
    Legacy wrapper - loads image to get dimensions.
    Prefer resolve_bbox_from_box_array() when image is already loaded.
    """
    img = ds["images"][idx].numpy()
    h, w = img.shape[:2]
    box = ds["boxes"][idx].numpy()
    return resolve_bbox_from_box_array(box, h, w)


def apply_bbox_crop_optimized(
    img: np.ndarray, 
    box: np.ndarray, 
    padding_ratio: float = 0.15
) -> np.ndarray:
    """
    OPTIMIZED: Crop to bounding box with padding.
    Takes pre-loaded image and box array to avoid redundant DeepLake access.
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


def aspect_preserving_resize(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resize image to target_size while preserving aspect ratio.
    Pads with reflection to make a square image.
    
    Args:
        img: Input image (H, W, C), uint8, RGB.
        target_size: Output size (target_size x target_size).
    
    Returns:
        Square image with preserved aspect ratio and reflection padding.
    """
    h, w = img.shape[:2]
    
    # Scale so longest edge = target_size
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    # Pad to square using reflection
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_REFLECT_101
    )
    
    return padded


# Label index utilities

def build_label_index(ds) -> Dict[int, np.ndarray]:
    """
    Build a mapping from class label -> array of dataset indices.
    Useful for creating few-shot subsets.
    """
    label_to_idxs: Dict[int, list] = defaultdict(list)
    for i, sample in tqdm(enumerate(ds), total=len(ds), desc="Building label index"):
        label = int(sample["labels"].numpy()[0])
        label_to_idxs[label].append(int(i))
    return {k: np.array(v, dtype=np.int64) for k, v in label_to_idxs.items()}


def save_label_index(label_index: Dict[int, np.ndarray], path):
    """Persist the label index to disk."""
    np.savez_compressed(path, **{str(k): v for k, v in label_index.items()})


def load_label_index(path) -> Dict[int, np.ndarray]:
    """Load a label index saved by save_label_index."""
    data = np.load(path)
    return {int(k): data[k] for k in data.files}



# Data splitting functions


def create_fewshot_split(
    label_index: Dict[int, np.ndarray],
    n_support: int = 5,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Split training dataset indices into four sets:
    - support_indices: Few-shot training set (5 images per class)
    - pool_indices: Unlabeled pool for pseudo-labeling
    - val_indices: Validation set (for monitoring during training)
    - test_indices: Test set from training data
    
    NOTE: The original ds_val is kept completely untouched as the final test set
    per project requirements.
    
    Args:
        label_index: Dict mapping class_id -> array of dataset indices
        n_support: Number of support samples per class (default 5)
        val_fraction: Fraction of remaining data for validation
        test_fraction: Fraction of remaining data for test
        seed: Random seed for reproducibility
    
    Returns:
        support_indices: Dict[class_id -> array of support indices]
        pool_indices: Dict[class_id -> array of unlabeled pool indices]
        val_indices: Dict[class_id -> array of validation indices]
        test_indices: Dict[class_id -> array of test indices]
    """
    rng = np.random.default_rng(seed)
    
    support_indices = {}
    pool_indices = {}
    val_indices = {}
    test_indices = {}
    
    for class_id, indices in label_index.items():
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        
        n_total = int(len(indices))
        if n_total == 0:
            support_indices[class_id] = np.array([], dtype=np.int64)
            pool_indices[class_id] = np.array([], dtype=np.int64)
            val_indices[class_id] = np.array([], dtype=np.int64)
            test_indices[class_id] = np.array([], dtype=np.int64)
            continue
        
        # Prioritize support (few-shot) first; val/test are for monitoring and may be empty
        # for very small classes.
        n_support_actual = min(int(n_support), n_total)
        remaining = n_total - n_support_actual
        
        n_val = 0
        n_test = 0
        if remaining > 0:
            # Compute requested sizes, then cap to what's remaining.
            if val_fraction > 0:
                n_val = max(1, int(round(n_total * val_fraction)))
            if test_fraction > 0:
                n_test = max(1, int(round(n_total * test_fraction)))
            n_val = min(n_val, remaining)
            remaining -= n_val
            n_test = min(n_test, remaining)
            remaining -= n_test
        
        # Split order: support, val, test, pool (rest)
        support_indices[class_id] = indices[:n_support_actual].copy()
        val_indices[class_id] = indices[n_support_actual:n_support_actual + n_val].copy()
        test_indices[class_id] = indices[n_support_actual + n_val:n_support_actual + n_val + n_test].copy()
        pool_indices[class_id] = indices[n_support_actual + n_val + n_test:].copy()
    
    return support_indices, pool_indices, val_indices, test_indices


def flatten_indices(indices_dict: Dict[int, np.ndarray]) -> np.ndarray:
    """Flatten a dict of class->indices to a single array."""
    all_indices = []
    for indices in indices_dict.values():
        all_indices.extend(indices)
    return np.array(all_indices, dtype=np.int64)


def get_labels_for_indices(ds, indices: np.ndarray) -> np.ndarray:
    """Get labels for a set of indices."""
    indices_list = [int(i) for i in indices]
    labels_np = ds["labels"][indices_list].numpy().astype(int)
    return labels_np.reshape(len(labels_np), -1)[:, 0]



# MultiBackboneFeatureExtractor

class MultiBackboneFeatureExtractor:
    """
    Feature extractor supporting multiple backbone architectures.
    
    Supported backbones:
    - 'resnet50': ResNet-50 (2048-dim embeddings)
    - 'efficientnet_b4': EfficientNet-B4 (1792-dim embeddings)
    - 'vit_b_16': Vision Transformer B/16 (768-dim embeddings)
    
    Preprocessing modes:
    - 'native': Backbone's ImageNet transforms (optionally pad-to-square first)
    - 'bbox_crop': Bbox crop (+padding) then backbone transforms (optionally pad-to-square first)
    """
    
    SUPPORTED_BACKBONES = ['resnet50', 'efficientnet_b4', 'vit_b_16']
    SUPPORTED_PREPROCESS_MODES = ['native', 'bbox_crop']
    
    def __init__(
        self,
        backbone_name: str,
        device: torch.device,
        preprocess_mode: str = 'native',
        pad_to_square: bool = True,
        bbox_padding_ratio: float = 0.15,
    ):
        if backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone must be one of {self.SUPPORTED_BACKBONES}")
        if preprocess_mode not in self.SUPPORTED_PREPROCESS_MODES:
            raise ValueError(f"Preprocess mode must be one of {self.SUPPORTED_PREPROCESS_MODES}")
        
        self.backbone_name = backbone_name
        self.device = device
        self.preprocess_mode = preprocess_mode
        self.pad_to_square = bool(pad_to_square)
        self.bbox_padding_ratio = float(bbox_padding_ratio)
        
        # Load the appropriate backbone
        if backbone_name == 'resnet50':
            self._init_resnet50()
        elif backbone_name == 'efficientnet_b4':
            self._init_efficientnet_b4()
        elif backbone_name == 'vit_b_16':
            self._init_vit_b_16()
        
        self.model = self.model.to(device).eval()
        pad_tag = "pad" if self.pad_to_square else "no-pad"
        print(f"Loaded {backbone_name} | mode={preprocess_mode} | {pad_tag} | dim={self.embedding_dim}")

    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
        import cv2

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
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )
    
    def _init_resnet50(self):
        """Initialize ResNet-50 backbone."""
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Identity()
        self.preprocess = weights.transforms()
        self.embedding_dim = 2048
    
    def _init_efficientnet_b4(self):
        """Initialize EfficientNet-B4 backbone."""
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b4(weights=weights)
        self.model.classifier = nn.Identity()
        self.preprocess = weights.transforms()
        self.embedding_dim = 1792
    
    def _init_vit_b_16(self):
        """Initialize Vision Transformer B/16 backbone."""
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = models.vit_b_16(weights=weights)
        self.model.heads = nn.Identity()
        self.preprocess = weights.transforms()
        self.embedding_dim = 768
    
    def _apply_preprocessing(self, img: np.ndarray, ds=None, idx: int = None) -> np.ndarray:
        """Apply preprocessing based on mode."""
        if self.preprocess_mode == 'bbox_crop' and ds is not None and idx is not None:
            img = apply_bbox_crop(img, ds, idx, padding_ratio=self.bbox_padding_ratio)
        if self.pad_to_square:
            img = self._pad_to_square(img)
        return img
    
    @torch.no_grad()
    def extract_single(self, image: np.ndarray, ds=None, idx: int = None) -> np.ndarray:
        """Extract embedding for a single image (H,W,C numpy array)."""
        image = self._apply_preprocessing(image, ds, idx)
        pil_img = Image.fromarray(image)
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of images (preprocessing already applied)."""
        tensors = torch.stack([
            self.preprocess(Image.fromarray(img)) for img in images
        ]).to(self.device)
        
        if self.device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                embeddings = self.model(tensors)
        else:
            embeddings = self.model(tensors)
        
        return embeddings.float().cpu().numpy()
    
    @torch.no_grad()
    def extract_from_dataset(
        self, 
        ds, 
        indices: np.ndarray, 
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract embeddings for specific dataset indices (batched)."""
        all_embeddings = []
        iterator = range(0, len(indices), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Extracting [{self.backbone_name}|{self.preprocess_mode}]")
        
        for i in iterator:
            batch_indices = [int(j) for j in indices[i:i+batch_size]]
            images_np = ds["images"][batch_indices].numpy(aslist=True)
            
            # Apply preprocessing (bbox crop + optional pad-to-square)
            images = [
                self._apply_preprocessing(img, ds, idx)
                for img, idx in zip(images_np, batch_indices)
            ]
            
            embeddings = self.extract_batch(images)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)




# visualize preprocessing 

def visualize_preprocessing_modes(
    ds,
    indices: List[int],
    backbone_name: str = "resnet50",
    padding_ratio: float = 0.15,
    pad_to_square: bool = True,
    figsize=(18, 12),
):
    """
    Visualize preprocessing for a given backbone, showing the *actual* images fed into the model.
    
    Columns:
      Original | Native (pad-to-square → weights.transforms()) | Bbox crop (pad) → pad-to-square → weights.transforms() | Bbox on Original
    """
    from PIL import Image
    from torchvision import models
    import cv2

    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    elif backbone_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    elif backbone_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    else:
        raise ValueError("backbone_name must be one of: resnet50, efficientnet_b4, vit_b_16")

    preprocess = weights.transforms()
    mean_vals = getattr(preprocess, "mean", None) or weights.meta.get("mean", [0.0, 0.0, 0.0])
    std_vals = getattr(preprocess, "std", None) or weights.meta.get("std", [1.0, 1.0, 1.0])
    mean = torch.tensor(mean_vals).view(-1, 1, 1)
    std = torch.tensor(std_vals).view(-1, 1, 1)

    # Build a display transform that matches weights.transforms() geometry but avoids normalization,
    # so colors look natural while still showing the exact crop/resize the model sees.
    display_preprocess = None
    try:
        from torchvision import transforms as T

        if hasattr(preprocess, "transforms"):
            geom_only = []
            for tr in preprocess.transforms:
                if isinstance(tr, T.Normalize):
                    continue
                geom_only.append(tr)
            display_preprocess = T.Compose(geom_only)
    except Exception:
        display_preprocess = None

    def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
        # DeepLake may return uint8 or float; PIL + torchvision expect uint8 RGB.
        if img.dtype != np.uint8:
            img = img.astype(np.float32, copy=False)
            if img.max() <= 1.0:
                img = (img * 255.0).round()
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _pad_to_square(img: np.ndarray) -> np.ndarray:
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
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )

    def _tensor_to_display(t: torch.Tensor) -> np.ndarray:
        # Some torchvision versions include Normalize in weights.transforms(), others don't.
        # Only undo normalization when the tensor looks normalized.
        t_cpu = t.detach().cpu().float()
        if t_cpu.min() < -0.05 or t_cpu.max() > 1.05:
            img_t = (t_cpu * std + mean)
        else:
            img_t = t_cpu
        img_t = img_t.clamp(0, 1)
        return img_t.permute(1, 2, 0).numpy()

    def _preprocess_for_display(pil_img: Image.Image) -> np.ndarray:
        if display_preprocess is not None:
            out = display_preprocess(pil_img)
            if isinstance(out, torch.Tensor):
                t_cpu = out.detach().cpu().float()
                # Handle PILToTensor-style 0..255 output
                if t_cpu.max() > 1.5:
                    t_cpu = t_cpu / 255.0
                t_cpu = t_cpu.clamp(0, 1)
                return t_cpu.permute(1, 2, 0).numpy()
            arr = _to_uint8_rgb(np.asarray(out))
            return (arr.astype(np.float32) / 255.0)

        # Fallback: apply full preprocess then de-normalize for display.
        t = preprocess(pil_img)
        return _tensor_to_display(t)
    n_samples = len(indices)
    
    fig, axes = plt.subplots(n_samples, 4, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Column headers
    col_titles = [
        'Original',
        f'Native ({backbone_name})',
        f'Bbox crop → native ({backbone_name})',
        'Bbox on Original',
    ]
    
    for row, idx in enumerate(indices):
        # Load original image
        img = _to_uint8_rgb(ds["images"][idx].numpy())
        bbox = resolve_bbox_xywh_or_xyxy(ds, idx)
        
        # Get class name
        class_id = int(ds["labels"][idx].numpy().item())
        
        # Col 0: Original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original\n{img.shape[1]}×{img.shape[0]}", fontsize=9)
        axes[row, 0].axis('off')
        
        # Col 1: After backbone-native preprocessing (exactly as fed into the model)
        h, w = img.shape[:2]
        img_native_geom = _pad_to_square(img) if pad_to_square else img
        native_img = _preprocess_for_display(Image.fromarray(img_native_geom))
        axes[row, 1].imshow(native_img)
        axes[row, 1].set_title(f"Native\n{native_img.shape[1]}×{native_img.shape[0]}", fontsize=9)
        axes[row, 1].axis('off')
        
        # Col 2: After bbox crop + backbone-native preprocessing (exactly as fed into the model)
        img_bbox = apply_bbox_crop(img, ds, idx, padding_ratio=padding_ratio)
        img_bbox = _to_uint8_rgb(img_bbox)
        img_bbox_geom = _pad_to_square(img_bbox) if pad_to_square else img_bbox
        bbox_img = _preprocess_for_display(Image.fromarray(img_bbox_geom))
        axes[row, 2].imshow(bbox_img)
        axes[row, 2].set_title(f"Bbox→Native\n{bbox_img.shape[1]}×{bbox_img.shape[0]}", fontsize=9)
        axes[row, 2].axis('off')
        
        # Col 3: Original with bbox overlay
        axes[row, 3].imshow(img)
        coverage = 0.0
        if bbox is not None:
            x1, y1, x2, y2 = map(float, bbox)
            coverage = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (h * w) * 100.0
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  fill=False, edgecolor='lime', linewidth=2)
            axes[row, 3].add_patch(rect)
            # Also show padded region
            box_w, box_h = x2 - x1, y2 - y1
            pad_x, pad_y = int(box_w * padding_ratio), int(box_h * padding_ratio)
            rect_padded = plt.Rectangle(
                (max(0, x1-pad_x), max(0, y1-pad_y)), 
                min(w, x2+pad_x) - max(0, x1-pad_x),
                min(h, y2+pad_y) - max(0, y1-pad_y),
                fill=False, edgecolor='yellow', linewidth=1, linestyle='--')
            axes[row, 3].add_patch(rect_padded)
        axes[row, 3].set_title(f"Bbox overlay\nCoverage: {coverage:.1f}%", fontsize=9)
        axes[row, 3].axis('off')
        
        # Row label
        axes[row, 0].set_ylabel(f"Class {class_id}\nIdx {idx}", fontsize=9, rotation=0, 
                                 ha='right', va='center', labelpad=40)
    
    # Set column titles
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title + '\n' + ax.get_title().split('\n')[-1], fontsize=10, fontweight='bold')
    
    plt.suptitle("Preprocessing Comparison: Native Center Crop vs Bounding Box Crop", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Preprocessing Summary:")
    print(f"  • Backbone: {backbone_name}")
    print(f"  • Pad-to-square before transforms: {pad_to_square}")
    print("  • Display: de-normalized for visualization")
    print("  • Native: (pad-to-square →) weights.transforms()")
    print(f"  • Bbox Crop: bbox crop (+{int(padding_ratio*100)}% padding) → (pad-to-square →) weights.transforms()")
    print(f"\n  Green box = tight bbox | Yellow dashed = bbox + {int(padding_ratio*100)}% padding")



def evaluate_backbone_fewshot(
    backbone_name: str,
    preprocess_mode: str,                 # NEW: 'native' or 'bbox_crop'
    ds: Any,
    val_indices: np.ndarray,
    val_labels: np.ndarray,
    support_indices: Dict[int, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
    max_val_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None
) -> Dict:
    """
    Evaluate a backbone + preprocessing mode for few-shot classification.
    """
    import time
    
    # If you change preprocessing logic, bump this to avoid reusing stale caches.
    PREPROCESS_IMPL_VERSION = "v3_bboxcrop_pad_to_square_then_native_transforms"
    config_name = f"{backbone_name}_{preprocess_mode}"
    print(f"\n{'='*60}")
    print(f"EVALUATING: {config_name.upper()}")
    print(f"{'='*60}")
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    extractor = MultiBackboneFeatureExtractor(backbone_name, device, preprocess_mode)
    
    # Cache setup - include preprocess_mode in filename
    cache_path_val = None
    cache_path_support = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path_val = cache_dir / f"{config_name}_val_cache.npz"
        cache_path_support = cache_dir / f"{config_name}_support_cache.npz"
    
    # 1. Compute prototypes from support set
    print("\nComputing class prototypes from support set...")
    class_ids_list_all = sorted(support_indices.keys())

    # Build a deterministic, flattened support set for batched embedding extraction.
    # Drop classes with zero support examples (otherwise they'd create all-zero prototypes).
    support_indices_flat = []
    support_class_ids_flat = []
    class_ids_list = []
    for class_id in class_ids_list_all:
        idxs = np.asarray(support_indices[class_id], dtype=int)
        if idxs.size == 0:
            continue
        class_ids_list.append(int(class_id))
        idxs = np.sort(idxs)
        support_indices_flat.extend(idxs.tolist())
        support_class_ids_flat.extend([int(class_id)] * len(idxs))

    if len(class_ids_list) == 0:
        raise ValueError("Support set is empty: no classes have support samples.")

    n_classes = len(class_ids_list)
    support_indices_flat = np.asarray(support_indices_flat, dtype=int)
    support_class_ids_flat = np.asarray(support_class_ids_flat, dtype=int)
    class_ids_arr = np.asarray(class_ids_list, dtype=int)
    
    # Check support cache
    support_cache_valid = False
    if cache_path_support and cache_path_support.exists():
        try:
            with np.load(cache_path_support) as cached:
                cached_ver = cached['preprocess_impl_version'].item() if 'preprocess_impl_version' in cached else None
                cached_class_ids = cached['class_ids'] if 'class_ids' in cached else None
                cached_support_idx = cached['support_indices_flat'] if 'support_indices_flat' in cached else None
                cached_support_cids = cached['support_class_ids_flat'] if 'support_class_ids_flat' in cached else None
                cached_dim = int(cached['embedding_dim']) if 'embedding_dim' in cached else None
                
                ver_match = cached_ver == PREPROCESS_IMPL_VERSION
                dim_match = cached_dim == extractor.embedding_dim
                class_ids_match = cached_class_ids is not None and np.array_equal(cached_class_ids, class_ids_arr)
                support_match = (
                    cached_support_idx is not None and cached_support_cids is not None and
                    np.array_equal(cached_support_idx, support_indices_flat) and
                    np.array_equal(cached_support_cids, support_class_ids_flat)
                )
                
                if ver_match and dim_match and class_ids_match and support_match:
                    prototypes = cached['prototypes']
                    shape_ok = (prototypes.shape[0] == n_classes and prototypes.shape[1] == extractor.embedding_dim)
                    if shape_ok:
                        support_cache_valid = True
                        print(f"  ✓ Support cache valid: loaded {len(prototypes)} prototypes")
                    else:
                        print("  ⚠️ Support cache invalid: prototype shape mismatch")
        except Exception as e:
            print(f"  ⚠️ Support cache load failed: {e}")
    
    if not support_cache_valid:
        sums = np.zeros((n_classes, extractor.embedding_dim), dtype=np.float32)
        counts = np.zeros(n_classes, dtype=np.int64)
        
        if support_indices_flat.size > 0:
            support_embeddings = extractor.extract_from_dataset(
                ds, support_indices_flat, batch_size, show_progress=False
            )
            class_to_row = {int(cid): i for i, cid in enumerate(class_ids_list)}
            for emb, cid in zip(support_embeddings, support_class_ids_flat):
                row = class_to_row[int(cid)]
                sums[row] += emb.astype(np.float32, copy=False)
                counts[row] += 1
        
        prototypes = np.zeros_like(sums, dtype=np.float32)
        nonzero = counts > 0
        prototypes[nonzero] = sums[nonzero] / counts[nonzero, None]
        if not np.all(nonzero):
            missing = class_ids_arr[~nonzero]
            raise ValueError(
                f"Failed to build prototypes for classes with zero support counts: {missing.tolist()}"
            )
        
        if cache_path_support:
            np.savez(cache_path_support, 
                     prototypes=prototypes,
                     embedding_dim=extractor.embedding_dim,
                     class_ids=class_ids_arr,
                     support_indices_flat=support_indices_flat,
                     support_class_ids_flat=support_class_ids_flat,
                     preprocess_impl_version=PREPROCESS_IMPL_VERSION)
            print(f"  Cached prototypes to {cache_path_support}")
    
    
    # 2. Handle max_val_samples limit
    if max_val_samples is not None and len(val_indices) > max_val_samples:
        rng = np.random.default_rng(42)
        subset_idx = rng.choice(len(val_indices), max_val_samples, replace=False)
        current_val_indices = val_indices[subset_idx]
        current_val_labels = val_labels[subset_idx]
    else:
        current_val_indices = val_indices
        current_val_labels = val_labels
    
    # Ensure we only score on labels that exist in the support set
    n_val_dropped = 0
    val_mask = np.isin(current_val_labels, class_ids_arr)
    if not np.all(val_mask):
        n_val_dropped = int((~val_mask).sum())
        dropped_labels = np.unique(current_val_labels[~val_mask])
        print(
            f"  ⚠️ Dropping {n_val_dropped} val samples with labels not in support set: {dropped_labels.tolist()}"
        )
        current_val_indices = current_val_indices[val_mask]
        current_val_labels = current_val_labels[val_mask]
    if len(current_val_indices) == 0:
        raise ValueError("No validation samples remain after filtering to support classes.")

    n_val = len(current_val_indices)
    
    # 3. Extract validation embeddings
    print(f"\nExtracting validation embeddings ({n_val} samples)...")
    
    cache_valid = False
    if cache_path_val and cache_path_val.exists() and max_val_samples is None:
        try:
            with np.load(cache_path_val) as cached:
                cached_ver = cached['preprocess_impl_version'].item() if 'preprocess_impl_version' in cached else None
                cached_indices = cached['indices']
                cached_embeddings = cached['embeddings']
                
                indices_match = (
                    len(cached_indices) == len(current_val_indices) and
                    np.array_equal(cached_indices, current_val_indices)
                )
                dim_match = cached_embeddings.shape[1] == extractor.embedding_dim
                
                ver_match = cached_ver == PREPROCESS_IMPL_VERSION
                
                if indices_match and dim_match and ver_match:
                    print(f"  ✓ Val cache valid: indices match, dim={cached_embeddings.shape[1]}")
                    val_embeddings = cached_embeddings
                    cache_valid = True
                else:
                    if not indices_match:
                        print(f"  ⚠️ Cache invalid: indices changed")
                    if not dim_match:
                        print(f"  ⚠️ Cache invalid: dim mismatch")
                    if not ver_match:
                        print(f"  ⚠️ Cache invalid: preprocess version mismatch")
        except Exception as e:
            print(f"  ⚠️ Cache load failed: {e}")
    
    if not cache_valid:
        val_embeddings = extractor.extract_from_dataset(ds, current_val_indices, batch_size)
        if cache_path_val and max_val_samples is None:
            np.savez(cache_path_val, 
                     indices=current_val_indices, 
                     embeddings=val_embeddings,
                     embedding_dim=extractor.embedding_dim,
                     preprocess_impl_version=PREPROCESS_IMPL_VERSION)
            print(f"  Cached embeddings to {cache_path_val}")
    
    # 4. Classify using cosine similarity
    print("Classifying validation samples...")
    prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    val_emb_norm = val_embeddings / (np.linalg.norm(val_embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = val_emb_norm @ prototypes_norm.T
    
    # Softmax for confidence
    logits = similarities * 10
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    pred_indices = np.argmax(similarities, axis=1)
    predictions = class_ids_arr[pred_indices]
    confidences = np.max(probs, axis=1)
    
    # 5. Compute metrics
    accuracy = (predictions == current_val_labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        current_val_labels, predictions, average="weighted", zero_division=0
    )
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    results = {
        "backbone": backbone_name,
        "preprocess_mode": preprocess_mode,
        "config": config_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "embedding_dim": extractor.embedding_dim,
        "time_seconds": elapsed_time,
        "n_val_samples": n_val,
        "mean_confidence": confidences.mean(),
        "n_val_dropped": n_val_dropped,
    }
    
    print(f"\n📊 RESULTS for {config_name}:")
    print(f"   Accuracy:   {accuracy*100:.2f}%")
    print(f"   F1 Score:   {f1*100:.2f}%")
    print(f"   Time:       {elapsed_time:.1f}s")
    
    del extractor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results



