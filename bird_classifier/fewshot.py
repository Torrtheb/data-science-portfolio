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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


from torchvision import models, transforms

# initial utils 
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



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
    
    print("\n Preprocessing Summary:")
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
    
    print(f"\n RESULTS for {config_name}:")
    print(f"   Accuracy:   {accuracy*100:.2f}%")
    print(f"   F1 Score:   {f1*100:.2f}%")
    print(f"   Time:       {elapsed_time:.1f}s")
    
    del extractor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results



# FeatureExtractor utilities

class FeatureExtractor:
    """Wrapper for a pretrained backbone as a feature extractor.

    Preprocessing modes:
    - 'native': backbone ImageNet transforms (optionally pad-to-square first)
    - 'bbox_crop': apply_bbox_crop (+padding) then transforms (optionally pad-to-square first)
    """

    SUPPORTED_BACKBONES = ["resnet50", "efficientnet_b4", "vit_b_16"]
    SUPPORTED_PREPROCESS_MODES = ["native", "bbox_crop"]

    def __init__(
        self,
        device: torch.device,
        backbone_name: Optional[str] = None,
        preprocess_mode: Optional[str] = None,
        pad_to_square: bool = True,
        bbox_padding_ratio: float = 0.15,
    ):
        self.device = device

        if backbone_name is None:
            backbone_name = globals().get('BEST_BACKBONE', globals().get('CACHED_BEST_BACKBONE', 'efficientnet_b4'))
        if preprocess_mode is None:
            preprocess_mode = (
                globals().get('BEST_PREPROCESS_MODE', globals().get('CACHED_BEST_PREPROCESS_MODE', 'bbox_crop'))
            )

        if backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone must be one of {self.SUPPORTED_BACKBONES}")
        if preprocess_mode not in self.SUPPORTED_PREPROCESS_MODES:
            raise ValueError(f"Preprocess mode must be one of {self.SUPPORTED_PREPROCESS_MODES}")

        self.backbone_name = backbone_name
        self.preprocess_mode = preprocess_mode
        self.pad_to_square = bool(pad_to_square)
        self.bbox_padding_ratio = float(bbox_padding_ratio)

        print(
            f"Initializing FeatureExtractor with {backbone_name} | mode={preprocess_mode} | "
            f"{'pad' if self.pad_to_square else 'no-pad'}..."
        )

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.model = models.resnet50(weights=weights)
            self.model.fc = nn.Identity()
            self.preprocess = weights.transforms()
            self.embedding_dim = 2048
        elif backbone_name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
            self.model = models.efficientnet_b4(weights=weights)
            self.model.classifier = nn.Identity()
            self.preprocess = weights.transforms()
            self.embedding_dim = 1792
        elif backbone_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            self.model = models.vit_b_16(weights=weights)
            self.model.heads = nn.Identity()
            self.preprocess = weights.transforms()
            self.embedding_dim = 768

        self.model = self.model.to(device).eval()

    def _ensure_uint8(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            return arr
        max_val = float(arr.max()) if arr.size else 1.0
        if max_val <= 1.0 + 1e-6:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
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

    def _apply_preprocessing(self, img: np.ndarray, ds=None, idx: int = None) -> np.ndarray:
        img = self._ensure_uint8(img)
        if self.preprocess_mode == "bbox_crop" and ds is not None and idx is not None:
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
        if self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                embedding = self.model(tensor)
        else:
            embedding = self.model(tensor)
        return embedding.float().cpu().numpy().flatten()

    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of images (preprocessing already applied)."""
        tensors = torch.stack([self.preprocess(Image.fromarray(img)) for img in images]).to(self.device)
        if self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
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
        show_progress: bool = True,
    ) -> np.ndarray:
        """Extract embeddings for specific dataset indices (batched)."""
        all_embeddings = []
        iterator = range(0, len(indices), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Extracting [{self.backbone_name}|{self.preprocess_mode}]")
        for i in iterator:
            batch_indices = [int(j) for j in indices[i : i + batch_size]]
            images_np = ds["images"][batch_indices].numpy(aslist=True)
            images = [self._apply_preprocessing(img, ds, idx) for img, idx in zip(images_np, batch_indices)]
            embeddings = self.extract_batch(images)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)



def classify_embedding(
    embedding: np.ndarray,
    prototypes: np.ndarray,
    class_ids: np.ndarray,
    metric: str = "cosine"
) -> Tuple[int, float, np.ndarray]:
    """
    Classify an embedding using nearest prototype.
    
    Args:
        embedding: (D,) query embedding
        prototypes: (C, D) class prototypes
        class_ids: (C,) class IDs
        metric: "cosine" or "euclidean"
    
    Returns:
        predicted_class: class ID
        confidence: confidence score (softmax probability)
        distances: distances to all prototypes
    """
    if metric == "cosine":
        # Cosine similarity (higher = more similar)
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        prototypes_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        similarities = embedding_norm @ prototypes_norm.T
        
        # Convert to "distances" (lower = more similar)
        distances = -similarities
    else:
        # Euclidean distance
        distances = np.linalg.norm(prototypes - embedding, axis=1)
    
    # Softmax over negative distances to get probabilities
    logits = -distances * 10  # Temperature scaling
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    
    pred_idx = np.argmin(distances)
    predicted_class = class_ids[pred_idx]
    confidence = probs[pred_idx]
    
    return int(predicted_class), float(confidence), distances



# Fewshotexperiment

class FewShotExperiment:
    """
    Manages the iterative few-shot learning experiment with verification.
    
    KEY CHANGES from original:
    1. Uses validation/test splits from TRAINING data for intermediate evaluation
    2. Keeps ds_val (original validation) completely untouched for final evaluation
    3. Clearer separation between training pool and evaluation sets
    
    This implementation:
    1. Provides manual verification mode to inspect candidates before adding
    2. Tracks per-class performance to identify struggling classes
    3. Gives honest metrics about pseudo-labeling effectiveness
    4. Focuses on building a labeled dataset for subsequent fine-tuning
    """
    
    def __init__(
        self,
        ds_train,
        support_indices: Dict[int, np.ndarray],
        pool_indices: Dict[int, np.ndarray],
        val_indices: Dict[int, np.ndarray],
        test_indices: Dict[int, np.ndarray],
        extractor: FeatureExtractor,
        n_support: int,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        batch_size: int = 64,
        use_fp16_embeddings: bool = True,
    ):

        self.ds_train = ds_train
        self.extractor = extractor
        self.n_support = n_support
        self.batch_size = batch_size
        self.use_fp16_embeddings = use_fp16_embeddings

        self.seed = seed
        # Use provided cache_dir or default to data/embedding_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/embedding_cache")
        
        self.backbone_name = getattr(extractor, 'backbone_name', 'resnet50')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the passed indices directly (no re-creation)
        self.support_indices = {k: v.copy() for k, v in support_indices.items()}
        self.pool_indices = {k: v.copy() for k, v in pool_indices.items()}
        self.val_indices = val_indices
        self.test_indices = test_indices
                
        # Flatten validation and test indices for easy access
        self.val_flat = flatten_indices(self.val_indices)
        self.test_flat = flatten_indices(self.test_indices)
        
        # Compute validation labels from val_indices dict (keys are class IDs)
        val_labels_list = []
        for class_id, indices in self.val_indices.items():
            val_labels_list.extend([class_id] * len(indices))
        self._val_labels = np.array(val_labels_list, dtype=np.int64)
        
        # Track iteration history
        self.history = []
        self.iteration = 0
        
        # Track per-class statistics
        self.per_class_stats = {}
        
        # Load or compute embeddings for TRAINING data with progress tracking
        self._train_embeddings = self._load_or_compute_embeddings(
            ds_train, f"{self.backbone_name}_train", len(ds_train)
        )
        
        # Final test (ds_val) is intentionally NOT embedded here to avoid leakage.
        # Load it ONLY at the end via set_final_test_dataset(...).
        self.ds_final_test = None
        self._final_test_embeddings = None
        self._final_test_labels = None

        # Initial prototype computation (uses cached embeddings)
        self.prototypes, self.class_ids = self._compute_prototypes_from_cache()
        self.class_id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}
        
        # Compute initial per-class accuracy on VALIDATION split (not final test)
        self._update_per_class_stats()
        
        # Print split summary
        print(f"\n{'='*60}")
        print("FEW-SHOT EXPERIMENT INITIALIZED")
        print(f"{'='*60}")
        print(f"Support set: {self.get_support_count()} samples ({self.n_support} per class)")
        print(f"Unlabeled pool: {self.get_pool_count()} samples")
        print(f"Validation (from train): {len(self.val_flat)} samples")
        print(f"Test (from train): {len(self.test_flat)} samples")
        if self.ds_final_test is None:
            print("Final test (ds_val): NOT LOADED (will load at the end)")
        else:
            print(f"Final test (ds_val): {len(self.ds_final_test)} samples (UNTOUCHED)")
        print(f"{'='*60}")
    
    def _load_or_compute_embeddings(
        self, 
        dataset, 
        name: str, 
        n_samples: int
    ) -> np.ndarray:
        """Load embeddings from disk if they exist, otherwise compute and save."""
        cache_path = self.cache_dir / f"{name}_embeddings.npy"
        
        if cache_path.exists():
            print(f"⚡ Loading cached {name} embeddings...")
            embeddings = np.load(cache_path)
            # Convert float16 back to float32 for computation
            if embeddings.dtype == np.float16:
                embeddings = embeddings.astype(np.float32)
            print(f"   Loaded {len(embeddings)} embeddings from cache")
            return embeddings
        
        print(f"🔄 Computing {name} embeddings (one-time, will be cached)...")
        embeddings = self._extract_with_progress(dataset, n_samples, batch_size=self.batch_size)
        
        # Save as float16 to reduce disk usage by 50%
        if self.use_fp16_embeddings:
            np.save(cache_path, embeddings.astype(np.float16))
            print(f"💾 Saved as float16 to {cache_path}")
        else:
            np.save(cache_path, embeddings)
        
        return embeddings
    
    def _extract_with_progress(
        self, 
        dataset, 
        n_samples: int, 
        batch_size: int = 64
    ) -> np.ndarray:
        """Extract embeddings with a progress bar (batched DeepLake reads)."""
        all_embeddings = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), 
                         total=n_batches, 
                         desc="Extracting features"):
                end_idx = min(i + batch_size, n_samples)
                batch_indices = list(range(i, end_idx))
                images_np = dataset["images"][batch_indices].numpy(aslist=True)
                images = [self.extractor._apply_preprocessing(img, dataset, idx) for img, idx in zip(images_np, batch_indices)]
                embeddings = self.extractor.extract_batch(images)
                all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def set_final_test_dataset(self, ds_final_test, final_test_indices: Optional[np.ndarray] = None):
        """Attach the held-out FINAL TEST dataset (DeepLake val split).

        This is intentionally delayed to avoid accidentally using ds_val during development.

        Args:
            ds_final_test: DeepLake dataset for the final evaluation.
            final_test_indices: Optional array of dataset indices to keep (e.g. duplicates dropped in EDA).
        """
        self.ds_final_test = ds_final_test

        # Compute or load embeddings + labels for the *full* dataset (cache-friendly)
        self._final_test_embeddings = self._load_or_compute_embeddings(
            ds_final_test, f"{self.backbone_name}_final_test", len(ds_final_test)
        )

        # Load or compute labels
        final_test_labels_path = self.cache_dir / f"{self.backbone_name}_final_test_labels.npy"
        if final_test_labels_path.exists():
            self._final_test_labels = np.load(final_test_labels_path)
        else:
            labels_np = ds_final_test['labels'][:].numpy().astype(int)
            self._final_test_labels = labels_np.reshape(len(labels_np), -1)[:, 0]
            np.save(final_test_labels_path, self._final_test_labels)

        # Optionally filter to cleaned indices (duplicates dropped)
        if final_test_indices is not None:
            keep = np.asarray(final_test_indices, dtype=np.int64)
            self._final_test_embeddings = self._final_test_embeddings[keep]
            self._final_test_labels = self._final_test_labels[keep]

        return True

    def evaluate_on_final_test(self) -> Dict:
        """
        Evaluate current prototypes on the FINAL held-out test dataset (ds_val).
        
        This should ONLY be called at the very end of the experiment.
        Returns accuracy, precision, recall, and f1 on the final test set.
        """
        if self._final_test_embeddings is None or self._final_test_labels is None:
            raise ValueError(
                "Final test dataset not loaded. Call set_final_test_dataset(ds_val) first."
            )
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Use current prototypes to classify final test embeddings
        prototypes_norm = self.prototypes / (
            np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8
        )
        test_emb_norm = self._final_test_embeddings / (
            np.linalg.norm(self._final_test_embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = test_emb_norm @ prototypes_norm.T
        
        pred_indices = np.argmax(similarities, axis=1)
        predictions = self.class_ids[pred_indices]
        true_labels = self._final_test_labels
        
        accuracy = (predictions == true_labels).mean()
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'predictions': predictions,
            'true_labels': true_labels,
        }

    def clear_cache(self):
        """Clear cached embeddings from disk."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"Cleared cache directory: {self.cache_dir}")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_prototypes_from_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prototypes using cached embeddings."""
        class_ids = sorted(self.support_indices.keys())
        prototypes = []
        
        for class_id in class_ids:
            indices = self.support_indices[class_id]
            
            if len(indices) == 0:
                prototypes.append(np.zeros(self.extractor.embedding_dim))
                continue
            
            embeddings = self._train_embeddings[indices]
            prototype = embeddings.mean(axis=0)
            prototypes.append(prototype)
        
        return np.array(prototypes), np.array(class_ids)
    
    def _update_per_class_stats(self, results: Optional[Dict] = None):
        """Compute per-class accuracy on validation set (from training data)."""
        if results is None:
            results = self.evaluate_on_val()
        
        for class_id in self.class_ids:
            mask = results['true_labels'] == class_id
            if mask.sum() > 0:
                class_acc = (results['predictions'][mask] == class_id).mean()
                self.per_class_stats[class_id] = {
                    'accuracy': class_acc,
                    'support_size': len(self.support_indices.get(class_id, [])),
                    'pool_size': len(self.pool_indices.get(class_id, [])),
                    'val_count': mask.sum()
                }
    
    def get_support_count(self) -> int:
        return sum(len(v) for v in self.support_indices.values())
    
    def get_pool_count(self) -> int:
        return sum(len(v) for v in self.pool_indices.values())
    
    def evaluate_on_val(self) -> Dict:
        """
        Evaluate current model on VALIDATION split (from training data).
        Use this for monitoring during iterative pseudo-labeling.
        """
        return self._evaluate_on_indices(
            self.val_flat, 
            self._val_labels, 
            self._train_embeddings,
            desc="Evaluating on validation"
        )
    
    def evaluate_on_test(self) -> Dict:
        """
        Evaluate on TEST split (from training data).
        Use this for final evaluation of the pseudo-labeling strategy.
        """
        # Compute test labels from test_indices dict
        test_labels_list = []
        for class_id, indices in self.test_indices.items():
            test_labels_list.extend([class_id] * len(indices))
        test_labels = np.array(test_labels_list, dtype=np.int64)
        
        return self._evaluate_on_indices(
            self.test_flat,
            test_labels,
            self._train_embeddings,
            desc="Evaluating on test"
        )
    
    def _evaluate_on_indices(
        self, 
        indices: np.ndarray, 
        labels: np.ndarray,
        embeddings: np.ndarray,
        desc: str = "Evaluating"
    ) -> Dict:
        """Evaluate on a specific set of indices with all metrics."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # Get embeddings for these indices
        eval_embeddings = embeddings[indices]
        
        # Compute predictions using prototypes
        prototypes_norm = self.prototypes / (np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8)
        eval_emb_norm = eval_embeddings / (np.linalg.norm(eval_embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = eval_emb_norm @ prototypes_norm.T
        
        # Softmax for confidence
        logits = similarities * 10
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        pred_indices = np.argmax(similarities, axis=1)
        predictions = self.class_ids[pred_indices]
        confidences = np.max(probs, axis=1)
        
        accuracy = (predictions == labels).mean()
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'true_labels': labels,
            'confidences': confidences,
            'similarities': similarities
        }
    
    def get_high_confidence_predictions(
        self,
        threshold: float = 0.8,
        max_per_class: Optional[int] = None
    ) -> Dict[int, List[Dict]]:
        """
        Get high-confidence predictions from the unlabeled pool.
        
        Returns dict: class_id -> list of {idx, pred_class, confidence, true_class}
        """
        # Flatten pool indices
        pool_flat = flatten_indices(self.pool_indices)
        if len(pool_flat) == 0:
            return {}
        
        # Get embeddings and classify
        pool_embeddings = self._train_embeddings[pool_flat]
        
        prototypes_norm = self.prototypes / (np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8)
        pool_emb_norm = pool_embeddings / (np.linalg.norm(pool_embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = pool_emb_norm @ prototypes_norm.T
        
        # Softmax
        logits = similarities * 10
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        pred_indices = np.argmax(similarities, axis=1)
        predictions = self.class_ids[pred_indices]
        confidences = np.max(probs, axis=1)
        
        # Get true labels for evaluation
        true_labels = get_labels_for_indices(self.ds_train, pool_flat)
        
        # Filter by confidence threshold
        high_conf_mask = confidences >= threshold
        
        # Organize by predicted class
        results = defaultdict(list)
        for i, (is_high_conf, pred_class, conf, true_label) in enumerate(
            zip(high_conf_mask, predictions, confidences, true_labels)
        ):
            if is_high_conf:
                results[int(pred_class)].append({
                    'idx': int(pool_flat[i]),
                    'pred_class': int(pred_class),
                    'confidence': float(conf),
                    'true_class': int(true_label)
                })
        
        # Sort by confidence and limit per class
        for class_id in results:
            results[class_id].sort(key=lambda x: x['confidence'], reverse=True)
            if max_per_class is not None:
                results[class_id] = results[class_id][:max_per_class]
        
        return dict(results)
    
    def add_to_support(self, indices: List[int], class_id: int):
        """
        Add samples to the support set for a specific class.
        Removes them from the pool.
        """
        indices = np.array(indices, dtype=np.int64)
        
        # Add to support
        if class_id not in self.support_indices:
            self.support_indices[class_id] = indices
        else:
            self.support_indices[class_id] = np.concatenate([
                self.support_indices[class_id], indices
            ])
        
        # Remove from pool.
        # IMPORTANT: pool_indices are keyed by TRUE class, but pseudo-labeling adds samples
        # to support by (predicted) class_id. So we must remove by index globally.
        if len(indices) > 0:
            remove_arr = np.asarray(indices, dtype=np.int64)
            for pool_cid, pool_arr in list(self.pool_indices.items()):
                pool_arr = np.asarray(pool_arr, dtype=np.int64)
                if pool_arr.size == 0:
                    continue
                self.pool_indices[pool_cid] = pool_arr[~np.isin(pool_arr, remove_arr)]
        
        # Recompute prototypes
        self.prototypes, self.class_ids = self._compute_prototypes_from_cache()
        self.class_id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}
    def run_iteration(
        self,
        confidence_threshold: float = 0.8,
        max_per_class: int = 5,
        use_true_labels: bool = True
    ) -> Dict:
        """
        Run one iteration of pseudo-labeling.

        If use_true_labels=True, uses ground truth to filter correct predictions (simulation).
        If False, all high-confidence predictions are added (real scenario).

        Returns iteration statistics.
        """
        self.iteration += 1

        # Get high-confidence predictions
        high_conf = self.get_high_confidence_predictions(
            threshold=confidence_threshold,
            max_per_class=max_per_class
        )

        n_candidates = sum(len(v) for v in high_conf.values())
        n_added = 0
        n_correct = 0
        n_wrong = 0
        accepted_samples = []  # for analysis/plotting

        # Collect adds per class so we only recompute prototypes once per class
        to_add_by_class = defaultdict(list)

        for class_id, candidates in high_conf.items():
            for cand in candidates:
                pred_ok = (cand['pred_class'] == cand['true_class'])

                if use_true_labels and not pred_ok:
                    # Simulation: reject incorrect predictions
                    n_wrong += 1
                    continue

                # Real mode: add all; Simulation: only correct reach here
                to_add_by_class[int(class_id)].append(int(cand['idx']))
                n_added += 1
                if pred_ok:
                    n_correct += 1
                else:
                    n_wrong += 1

                accepted_samples.append({
                    'idx': int(cand['idx']),
                    'pred_class': int(cand['pred_class']),
                    'true_class': int(cand['true_class']),
                    'confidence': float(cand['confidence'])
                })

        for class_id, idxs in to_add_by_class.items():
            if idxs:
                self.add_to_support(idxs, class_id)

        # Evaluate after adding
        val_results = self.evaluate_on_val()
        self._update_per_class_stats(val_results)

        iteration_stats = {
            'iteration': self.iteration,
            'threshold_used': float(confidence_threshold),

            # candidates / adds
            'n_candidates': int(n_candidates),
            'n_added': int(n_added),
            'samples_added': int(n_added),
            'n_correct': int(n_correct),
            'n_wrong': int(n_wrong),
            'added_samples': accepted_samples,

            # sizes
            'support_count': int(self.get_support_count()),
            'support_size': int(self.get_support_count()),
            'pool_count': int(self.get_pool_count()),
            'pool_size': int(self.get_pool_count()),

            # metrics (validation split)
            'val_accuracy': float(val_results['accuracy']),
            'val_precision': float(val_results.get('precision', 0.0)),
            'val_recall': float(val_results.get('recall', 0.0)),
            'val_f1': float(val_results.get('f1', 0.0)),

            'accuracy_after': float(val_results['accuracy']),
            'precision_after': float(val_results.get('precision', 0.0)),
            'recall_after': float(val_results.get('recall', 0.0)),
            'f1_after': float(val_results.get('f1', 0.0)),
        }

        self.history.append(iteration_stats)
        return iteration_stats
    
    def get_worst_classes(self, n: int = 5) -> List[Tuple[int, float]]:
        """Get the n classes with lowest validation accuracy."""
        class_accs = [
            (cid, stats['accuracy']) 
            for cid, stats in self.per_class_stats.items()
        ]
        class_accs.sort(key=lambda x: x[1])
        return class_accs[:n]
    
    def print_status(self):
        """Print current experiment status."""
        print(f"\n{'='*50}")
        print(f"ITERATION {self.iteration} STATUS")
        print(f"{'='*50}")
        print(f"Support set: {self.get_support_count()} samples")
        print(f"Pool remaining: {self.get_pool_count()} samples")
        
        if self.history:
            latest = self.history[-1]
            acc_key = 'accuracy_after' if 'accuracy_after' in latest else 'val_accuracy'
            print(f"Val accuracy: {latest.get(acc_key, 0)*100:.2f}%")
            if 'precision_after' in latest:
                print(f"Precision: {latest['precision_after']*100:.2f}%")
                print(f"Recall: {latest['recall_after']*100:.2f}%")
                print(f"F1: {latest['f1_after']*100:.2f}%")
        
        print(f"{'='*50}")

    def plot_progress(self, figsize=(14, 5)):
        """
        Plot training progress curves showing metrics over iterations.
        Call this after each iteration to visualize improvement.
        """
        if not self.history:
            print("No history to plot. Run at least one iteration first.")
            return
        
        import matplotlib.pyplot as plt
        
        # Extract data from history
        iterations = [h['iteration'] for h in self.history]
        
        # Handle both old and new history formats
        accuracies = [h.get('accuracy_after', h.get('val_accuracy', 0)) for h in self.history]
        precisions = [h.get('precision_after', h.get('accuracy_after', 0)) for h in self.history]
        recalls = [h.get('recall_after', h.get('accuracy_after', 0)) for h in self.history]
        f1s = [h.get('f1_after', h.get('accuracy_after', 0)) for h in self.history]
        samples_added = [h.get('samples_added', h.get('n_added', 0)) for h in self.history]
        support_sizes = [h.get('support_size', h.get('support_count', 0)) for h in self.history]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: All metrics
        ax1 = axes[0]
        ax1.plot(iterations, [a*100 for a in accuracies], 'b-o', label='Accuracy', linewidth=2, markersize=8)
        ax1.plot(iterations, [p*100 for p in precisions], 'g-s', label='Precision', linewidth=2, markersize=6)
        ax1.plot(iterations, [r*100 for r in recalls], 'r-^', label='Recall', linewidth=2, markersize=6)
        ax1.plot(iterations, [f*100 for f in f1s], 'm-d', label='F1 Score', linewidth=2, markersize=6)
        ax1.axhline(y=70, color='gray', linestyle='--', alpha=0.7, label='Target 70%')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score (%)')
        ax1.set_title('Metrics Over Iterations')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot 2: Samples added per iteration
        ax2 = axes[1]
        ax2.bar(iterations, samples_added, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Samples Added')
        ax2.set_title('Samples Added Per Iteration')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Support set growth
        ax3 = axes[2]
        ax3.plot(iterations, support_sizes, 'g-o', linewidth=2, markersize=8)
        ax3.fill_between(iterations, 0, support_sizes, alpha=0.2, color='green')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Support Set Size')
        ax3.set_title('Support Set Growth')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Few-Shot Pseudo-Labeling Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print summary
        if len(accuracies) > 1:
            improvement = (accuracies[-1] - accuracies[0]) * 100
            print(f"\nProgress: {accuracies[0]*100:.2f}% → {accuracies[-1]*100:.2f}% ({improvement:+.2f}%)")
            print(f"Total samples added: {sum(samples_added)}")

    def get_final_support_indices(self) -> set:
        """Get all indices currently in the support set."""
        all_indices = set()
        for class_id, indices in self.support_indices.items():
            all_indices.update(int(i) for i in indices)
        return all_indices

    def analyze_per_class_performance(self, top_n: int = 10):
        """Analyze and print per-class performance on validation set."""
        results = self.evaluate_on_val()
        predictions = results['predictions']
        true_labels = results['true_labels']
        
        # Compute per-class accuracy
        class_stats = {}
        for class_id in np.unique(true_labels):
            mask = true_labels == class_id
            if mask.sum() > 0:
                class_acc = (predictions[mask] == true_labels[mask]).mean()
                class_stats[int(class_id)] = {
                    'accuracy': float(class_acc),
                    'count': int(mask.sum()),
                    'support_size': len(self.support_indices.get(class_id, []))
                }
        
        # Sort by accuracy (worst first)
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['accuracy'])
        
        print(f"\n{'='*60}")
        print(f"WORST PERFORMING CLASSES (bottom {top_n})")
        print(f"{'='*60}")
        print(f"{'Class':>8} {'Accuracy':>10} {'Val Count':>10} {'Support':>10}")
        print("-" * 45)
        
        for class_id, stats in sorted_classes[:top_n]:
            print(f"{class_id:>8} {stats['accuracy']*100:>9.1f}% {stats['count']:>10} {stats['support_size']:>10}")
        
        # Also show best classes
        print(f"\n{'='*60}")
        print(f"BEST PERFORMING CLASSES (top {top_n})")
        print(f"{'='*60}")
        for class_id, stats in sorted_classes[-top_n:][::-1]:
            print(f"{class_id:>8} {stats['accuracy']*100:>9.1f}% {stats['count']:>10} {stats['support_size']:>10}")

    def get_pseudo_labeling_summary(self):
        """Print a summary of the pseudo-labeling process."""
        print(f"\n{'='*60}")
        print("PSEUDO-LABELING SUMMARY")
        print(f"{'='*60}")
        print(f"Total iterations: {self.iteration}")
        print(f"Support set size: {self.get_support_count()}")
        print(f"Pool remaining: {self.get_pool_count()}")
        
        if self.history:
            total_added = sum(h.get('samples_added', h.get('n_added', 0)) for h in self.history)
            print(f"Total samples added: {total_added}")
            
            first_acc = self.history[0].get('accuracy_after', self.history[0].get('val_accuracy', 0))
            last_acc = self.history[-1].get('accuracy_after', self.history[-1].get('val_accuracy', 0))
            print(f"\nAccuracy improvement: {first_acc*100:.2f}% → {last_acc*100:.2f}% ({(last_acc-first_acc)*100:+.2f}%)")
        print(f"{'='*60}")

    def print_metrics_summary(self):
        """Print a table of metrics for all iterations."""
        if not self.history:
            print("No iterations completed yet.")
            return
        
        print(f"\n{'='*80}")
        print("ITERATION METRICS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Iter':>5} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Added':>8}")
        print("-" * 60)
        
        for h in self.history:
            iteration = h.get('iteration', 0)
            acc = h.get('accuracy_after', h.get('val_accuracy', 0))
            prec = h.get('precision_after', acc)
            rec = h.get('recall_after', acc)
            f1 = h.get('f1_after', acc)
            added = h.get('samples_added', h.get('n_added', 0))
            print(f"{iteration:>5} {acc*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}% {added:>8}")
        print(f"{'='*80}")
    def run_auto_pseudo_labeling(
        self,
        target_accuracy: float = 0.70,
        initial_threshold: float = 0.8,
        min_threshold: float = 0.2,
        threshold_decay: float = 0.1,
        max_iterations: int = 50,
        max_per_class: int = 5,
        use_true_labels: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Automatically run pseudo-labeling iterations until target accuracy is reached.

        Metrics (accuracy/precision/recall/f1) are evaluated on the TRAIN-SPLIT validation
        indices (from ds_train), never on ds_val.
        """
        print(f"\n{'='*70}")
        print("AUTOMATIC PSEUDO-LABELING")
        print(f"{'='*70}")
        print(f"Target accuracy: {target_accuracy*100:.0f}%")
        print(f"Initial threshold: {initial_threshold}")
        print(f"Min threshold: {min_threshold}")
        print(f"Max iterations: {max_iterations}")
        print(f"Max per class: {max_per_class}")
        print(f"Mode: {'Simulation (uses true labels)' if use_true_labels else 'Real (no true labels)'}")
        print(f"{'='*70}\n")

        current_threshold = float(initial_threshold)
        iterations_without_candidates = 0

        # Get initial metrics
        initial_results = self.evaluate_on_val()
        current_accuracy = float(initial_results['accuracy'])
        current_precision = float(initial_results.get('precision', 0.0))
        current_recall = float(initial_results.get('recall', 0.0))
        current_f1 = float(initial_results.get('f1', 0.0))

        print(
            f"Starting metrics | "
            f"Acc: {current_accuracy*100:.2f}% | "
            f"Prec: {current_precision*100:.2f}% | "
            f"Rec: {current_recall*100:.2f}% | "
            f"F1: {current_f1*100:.2f}%"
        )
        print(f"Starting support size: {self.get_support_count()}")
        print(f"Pool size: {self.get_pool_count()}\n")

        for _ in range(max_iterations):
            if current_accuracy >= target_accuracy:
                print(f"\n🎯 TARGET REACHED! Accuracy: {current_accuracy*100:.2f}%")
                break

            if self.get_pool_count() == 0:
                print(f"\n⚠️ Pool exhausted! No more samples to add.")
                break

            stats = self.run_iteration(
                confidence_threshold=current_threshold,
                max_per_class=max_per_class,
                use_true_labels=use_true_labels
            )

            current_accuracy = float(stats.get('accuracy_after', stats.get('val_accuracy', 0.0)))
            current_precision = float(stats.get('precision_after', stats.get('val_precision', 0.0)))
            current_recall = float(stats.get('recall_after', stats.get('val_recall', 0.0)))
            current_f1 = float(stats.get('f1_after', stats.get('val_f1', 0.0)))

            if verbose:
                print(
                    f"Iter {stats['iteration']:3d} | "
                    f"Thresh: {current_threshold:.2f} | "
                    f"Added: {stats['n_added']:3d} | "
                    f"Acc: {current_accuracy*100:.2f}% | "
                    f"F1: {current_f1*100:.2f}% | "
                    f"Support: {stats['support_count']} | "
                    f"Pool: {stats['pool_count']}"
                )

            no_progress = (stats['n_candidates'] == 0) or (use_true_labels and stats['n_added'] == 0)

            if no_progress:
                iterations_without_candidates += 1
                if current_threshold > min_threshold:
                    current_threshold = max(min_threshold, current_threshold - threshold_decay)
                    print(f"   → Lowering threshold to {current_threshold:.2f}")
                else:
                    print(f"   → At minimum threshold, no more candidates")
                    if iterations_without_candidates >= 3:
                        print("   → Stopping (3 consecutive iterations without candidates)")
                        break
            else:
                iterations_without_candidates = 0

        print(f"\n{'='*70}")
        print("AUTOMATIC PSEUDO-LABELING COMPLETE")
        print(f"{'='*70}")
        print(f"Iterations run: {self.iteration}")
        print(
            f"Final metrics | "
            f"Acc: {current_accuracy*100:.2f}% | "
            f"Prec: {current_precision*100:.2f}% | "
            f"Rec: {current_recall*100:.2f}% | "
            f"F1: {current_f1*100:.2f}%"
        )
        print(f"Final support size: {self.get_support_count()}")
        print(f"Pool remaining: {self.get_pool_count()}")

        total_added = sum(h.get('samples_added', h.get('n_added', 0)) for h in self.history)
        total_correct = sum(h.get('n_correct', 0) for h in self.history)
        total_wrong = sum(h.get('n_wrong', 0) for h in self.history)

        print(f"\nTotal samples added: {total_added}")
        if use_true_labels:
            print(f"  Correct: {total_correct}")
            print(f"  Wrong (rejected): {total_wrong}")

        if current_accuracy >= target_accuracy:
            print(f"\n✅ SUCCESS: Target accuracy of {target_accuracy*100:.0f}% reached!")
        else:
            print(f"\n⚠️ Target not reached. Current: {current_accuracy*100:.2f}%")

        print(f"{'='*70}")

        return {
            'final_accuracy': current_accuracy,
            'final_precision': current_precision,
            'final_recall': current_recall,
            'final_f1': current_f1,
            'iterations': self.iteration,
            'total_added': total_added,
            'target_reached': current_accuracy >= target_accuracy
        }



# ============================================
# MANUAL VERIFICATION WORKFLOW
# ============================================
# This provides an interactive way to manually verify
# pseudo-labeled samples before adding them to training data.

class ManualVerificationSession:
    """
    Interactive session for manually verifying pseudo-labeled candidates.
    
    Workflow:
    1. Get high-confidence predictions from the pool
    2. Display each candidate alongside support set examples
    3. User decides: Accept (a), Reject (r), or Skip (s)
    4. Accepted samples are added to support set
    
    Usage:
        session = ManualVerificationSession(experiment)
        session.start_verification(confidence_threshold=0.5, max_candidates=50)
        # ... interactive verification ...
        session.apply_verified_samples()
    """
    
    def __init__(self, experiment: FewShotExperiment):
        self.experiment = experiment
        self.candidates_to_review = []
        self.verified_samples = []  # Samples accepted by user
        self.rejected_samples = []  # Samples rejected by user
        self.skipped_samples = []   # Samples skipped by user
        
    def prepare_candidates(
        self,
        confidence_threshold: float = 0.5,
        max_per_class: int = 5,
        max_total: int = 100,
        prioritize_classes: Optional[List[int]] = None
    ) -> int:
        """
        Prepare candidates for manual review.
        
        Args:
            confidence_threshold: Minimum confidence for inclusion
            max_per_class: Maximum candidates per class
            max_total: Maximum total candidates to review
            prioritize_classes: If provided, prioritize these class IDs
            
        Returns:
            Number of candidates prepared for review
        """
        print(f"Finding candidates with confidence ≥ {confidence_threshold}...")
        
        high_conf = self.experiment.get_high_confidence_predictions(
            threshold=confidence_threshold,
            max_per_class=max_per_class
        )
        
        # Flatten and prepare for review
        self.candidates_to_review = []
        
        for class_id, candidates in high_conf.items():
            for cand in candidates:
                self.candidates_to_review.append({
                    "idx": cand["idx"],
                    "pred_class": cand["pred_class"],
                    "true_class": cand.get("true_class"),  # For evaluation only
                    "confidence": cand["confidence"],
                    "status": "pending"  # pending, accepted, rejected, skipped
                })
        
        # Sort by confidence (highest first)
        self.candidates_to_review.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Limit total
        if len(self.candidates_to_review) > max_total:
            self.candidates_to_review = self.candidates_to_review[:max_total]
        
        print(f"Prepared {len(self.candidates_to_review)} candidates for manual review")
        return len(self.candidates_to_review)
    
    def _display_candidate(
        self, 
        candidate: Dict, 
        candidate_num: int,
        total: int,
        show_support: bool = True,
        n_support_examples: int = 3
    ):
        """Display a single candidate for verification."""
        pred_class = candidate["pred_class"]
        
        # Get support examples for the predicted class
        support_indices = self.experiment.support_indices.get(pred_class, [])
        n_support_to_show = min(n_support_examples, len(support_indices))
        
        n_cols = 1 + n_support_to_show if show_support else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        
        if n_cols == 1:
            axes = [axes]
        
        # Display support examples first (if any)
        if show_support and n_support_to_show > 0:
            support_ids = [int(i) for i in support_indices[:n_support_to_show]]
            support_images = self.experiment.ds_train["images"][support_ids].numpy(aslist=True)
            support_images = [
                self.experiment.extractor._apply_preprocessing(img, self.experiment.ds_train, idx)
                for img, idx in zip(support_images, support_ids)
            ]
            
            for i, img in enumerate(support_images):
                axes[i].imshow(img)
                axes[i].set_title(f"SUPPORT\nClass {pred_class}", fontsize=11, color='green', fontweight='bold')
                axes[i].axis("off")
                # Green border
                for spine in axes[i].spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('green')
                    spine.set_linewidth(4)
            offset = n_support_to_show
        else:
            offset = 0
        
        # Display candidate
        cand_idx = int(candidate["idx"])
        cand_img = self.experiment.ds_train["images"][cand_idx].numpy()
        cand_img = self.experiment.extractor._apply_preprocessing(cand_img, self.experiment.ds_train, cand_idx)
        
        ax = axes[offset]
        ax.imshow(cand_img)
        ax.set_title(
            f"CANDIDATE #{candidate_num}/{total}\n"
            f"Predicted: Class {pred_class}\n"
            f"Confidence: {candidate['confidence']:.1%}",
            fontsize=11, color='blue', fontweight='bold'
        )
        ax.axis("off")
        # Blue border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('blue')
            spine.set_linewidth(4)
        
        plt.suptitle(
            f"Does this image belong to Class {pred_class}?",
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()
    
    def verify_single(self, candidate_num: int) -> Optional[str]:
        """
        Verify a single candidate interactively.
        
        Returns: 'accepted', 'rejected', 'skipped', or None if invalid
        """
        if candidate_num < 0 or candidate_num >= len(self.candidates_to_review):
            print(f"Invalid candidate number: {candidate_num}")
            return None
        
        candidate = self.candidates_to_review[candidate_num]
        
        if candidate["status"] != "pending":
            print(f"Candidate {candidate_num} already {candidate['status']}")
            return candidate["status"]
        
        # Display the candidate
        self._display_candidate(
            candidate, 
            candidate_num + 1, 
            len(self.candidates_to_review)
        )
        
        print("\n" + "="*50)
        print("DECISION:")
        print("  [a] ACCEPT - Add to training set as class", candidate["pred_class"])
        print("  [r] REJECT - Do not add (wrong prediction)")
        print("  [s] SKIP   - Unsure, skip for now")
        print("  [q] QUIT   - Stop verification session")
        print("="*50)
        
        while True:
            try:
                decision = input("Your decision (a/r/s/q): ").strip().lower()
            except EOFError:
                # Non-interactive environment
                print("Non-interactive environment detected. Use batch verification instead.")
                return None
            
            if decision == 'a':
                candidate["status"] = "accepted"
                self.verified_samples.append(candidate)
                print(f"✅ ACCEPTED - Will add to class {candidate['pred_class']}")
                return "accepted"
            elif decision == 'r':
                candidate["status"] = "rejected"
                self.rejected_samples.append(candidate)
                print("❌ REJECTED")
                return "rejected"
            elif decision == 's':
                candidate["status"] = "skipped"
                self.skipped_samples.append(candidate)
                print("⏭️ SKIPPED")
                return "skipped"
            elif decision == 'q':
                print("Quitting verification session...")
                return "quit"
            else:
                print("Invalid input. Please enter a, r, s, or q.")
    
    def start_verification(
        self,
        confidence_threshold: float = 0.5,
        max_candidates: int = 50,
        max_per_class: int = 5
    ):
        """
        Start an interactive verification session.
        
        This will display candidates one by one and ask for your decision.
        """
        # Prepare candidates
        n_candidates = self.prepare_candidates(
            confidence_threshold=confidence_threshold,
            max_per_class=max_per_class,
            max_total=max_candidates
        )
        
        if n_candidates == 0:
            print("No candidates found at this threshold. Try lowering it.")
            return
        
        print(f"\n{'='*60}")
        print("MANUAL VERIFICATION SESSION")
        print(f"{'='*60}")
        print(f"You will review {n_candidates} candidates.")
        print("For each candidate, compare it with support examples.")
        print("Decide if the predicted class is correct.\n")
        
        # Reset counters
        self.verified_samples = []
        self.rejected_samples = []
        self.skipped_samples = []
        
        # Verify each candidate
        for i, candidate in enumerate(self.candidates_to_review):
            if candidate["status"] != "pending":
                continue
            
            result = self.verify_single(i)
            
            if result == "quit":
                break
            
            # Show progress
            print(f"\nProgress: {i+1}/{n_candidates}")
            print(f"  Accepted: {len(self.verified_samples)}")
            print(f"  Rejected: {len(self.rejected_samples)}")
            print(f"  Skipped:  {len(self.skipped_samples)}")
            print("-" * 40)
        
        # Summary
        self._print_session_summary()
    
    def _print_session_summary(self):
        """Print summary of verification session."""
        print(f"\n{'='*60}")
        print("VERIFICATION SESSION COMPLETE")
        print(f"{'='*60}")
        print(f"  ✅ Accepted: {len(self.verified_samples)}")
        print(f"  ❌ Rejected: {len(self.rejected_samples)}")
        print(f"  ⏭️ Skipped:  {len(self.skipped_samples)}")
        
        pending = sum(1 for c in self.candidates_to_review if c["status"] == "pending")
        if pending > 0:
            print(f"  ⏳ Pending:  {pending}")
        
        # Show accuracy of predictions (using ground truth for evaluation)
        if self.verified_samples:
            correct_accepted = sum(1 for s in self.verified_samples 
                                   if s.get("true_class") == s["pred_class"])
            print(f"\n📊 Accuracy of accepted samples: {correct_accepted}/{len(self.verified_samples)}")
            print("   (Ground truth check - you wouldn't have this in production)")
        
        print(f"\n💡 Next step: Call session.apply_verified_samples() to add accepted samples")
        print(f"{'='*60}")
    
    def apply_verified_samples(self) -> Dict:
        """
        Add all accepted samples to the support set.
        
        Returns:
            Dict with summary of applied changes
        """
        if not self.verified_samples:
            print("No verified samples to apply.")
            return {"added": 0}
        
        print(f"\n{'='*60}")
        print("APPLYING VERIFIED SAMPLES")
        print(f"{'='*60}")
        
        # Convert to the format expected by add_to_support
        samples_to_add = []
        for sample in self.verified_samples:
            samples_to_add.append({
                "idx": sample["idx"],
                "pred_class": sample["pred_class"],
                "true_class": sample.get("true_class"),
                "confidence": sample["confidence"],
                "correct": sample.get("true_class") == sample["pred_class"]
            })
        
        # Add to support set
        added = self.experiment.add_to_support(samples_to_add)
        
        # Evaluate after adding
        results_after = self.experiment.evaluate_on_val()
        
        print(f"\n✅ Added {len(added)} samples to support set")
        print(f"   New support set size: {self.experiment.get_support_count()}")
        print(f"   Validation accuracy: {results_after['accuracy']*100:.2f}%")
        
        # Clear verified samples (they've been applied)
        self.verified_samples = []
        
        return {
            "added": len(added),
            "new_support_size": self.experiment.get_support_count(),
            "accuracy": results_after["accuracy"]
        }
    
    def get_status(self) -> Dict:
        """Get current session status."""
        return {
            "total_candidates": len(self.candidates_to_review),
            "accepted": len(self.verified_samples),
            "rejected": len(self.rejected_samples),
            "skipped": len(self.skipped_samples),
            "pending": sum(1 for c in self.candidates_to_review if c["status"] == "pending")
        }


# ============================================
# BATCH VERIFICATION (for non-interactive use)
# ============================================
# Use this if you want to review all candidates at once
# and make decisions in a separate cell

class BatchVerificationSession:
    """
    Batch verification for reviewing multiple candidates at once.
    
    Better for Jupyter notebook workflow where you can:
    1. Display a grid of candidates
    2. Enter decisions as a list
    3. Apply all at once
    
    Usage:
        batch = BatchVerificationSession(experiment)
        batch.prepare_and_display(confidence_threshold=0.5, n_candidates=20)
        # Look at the displayed grid
        batch.set_decisions([1, 1, 0, 1, 0, ...])  # 1=accept, 0=reject
        batch.apply_decisions()
    """
    
    def __init__(self, experiment: FewShotExperiment):
        self.experiment = experiment
        self.candidates = []
        self.decisions = []
        self.last_confidence_threshold = None
    
    def prepare_and_display(
        self,
        confidence_threshold: float = 0.5,
        n_candidates: int = 20,
        max_per_class: int = 5,
        show_support: bool = True,
        reveal_ground_truth: bool = False
    ):
        """
        Prepare candidates and display them grouped by predicted class.
        Each row shows candidates for a single predicted class.
        """
        self.last_confidence_threshold = float(confidence_threshold)
        print(f"Finding candidates with confidence ≥ {confidence_threshold}...")
        
        high_conf = self.experiment.get_high_confidence_predictions(
            threshold=confidence_threshold,
            max_per_class=max_per_class
        )
        
        if not high_conf:
            print("No candidates found at this threshold.")
            return
        
        # Group candidates by predicted class, limit total
        # Sort classes by highest confidence candidate
        class_order = sorted(
            high_conf.keys(),
            key=lambda cid: max(c["confidence"] for c in high_conf[cid]),
            reverse=True
        )
        
        # Build ordered list of candidates (grouped by class)
        self.candidates = []
        candidates_by_class = {}
        total_added = 0
        
        for class_id in class_order:
            if total_added >= n_candidates:
                break
            cands = sorted(high_conf[class_id], key=lambda x: x["confidence"], reverse=True)
            class_cands = []
            for c in cands:
                if total_added >= n_candidates:
                    break
                c["display_idx"] = len(self.candidates)  # Track original index for decisions
                self.candidates.append(c)
                class_cands.append(c)
                total_added += 1
            if class_cands:
                candidates_by_class[class_id] = class_cands
        
        if not self.candidates:
            print("No candidates found at this threshold.")
            return
        
        print(f"Displaying {len(self.candidates)} candidates across {len(candidates_by_class)} classes: \n")
        
        # Display: one row per class, +1 column for reference image
        n_rows = len(candidates_by_class)
        max_cands = max(len(cands) for cands in candidates_by_class.values())
        n_cols = max_cands + 1  # +1 for reference support image
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3.5*n_rows), squeeze=False)
        
        for row_idx, class_id in enumerate(candidates_by_class.keys()):
            cands = candidates_by_class[class_id]
            
            # First column: Reference support image for this class
            ax_ref = axes[row_idx, 0]
            support_indices = self.experiment.support_indices.get(class_id, [])
            if len(support_indices) > 0:
                ref_idx = int(support_indices[0])
                ref_img = self.experiment.ds_train["images"][ref_idx].numpy()
                ref_img = self.experiment.extractor._apply_preprocessing(ref_img, self.experiment.ds_train, ref_idx)
                ax_ref.imshow(ref_img)
            ax_ref.set_xlabel(f"Class {class_id}\n(REFERENCE)", fontsize=9, fontweight='bold', color='green')
            ax_ref.set_xticks([])
            ax_ref.set_yticks([])
            for spine in ax_ref.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('green')
                spine.set_linewidth(4)
            
            # Remaining columns: candidates
            for col_idx in range(max_cands):
                ax = axes[row_idx, col_idx + 1]  # +1 because first col is reference
                
                if col_idx < len(cands):
                    cand = cands[col_idx]
                    
                    # Load image
                    idx_int = int(cand["idx"])
                    img = self.experiment.ds_train["images"][idx_int].numpy()
                    img = self.experiment.extractor._apply_preprocessing(img, self.experiment.ds_train, idx_int)
                    ax.imshow(img)
                    
                    # Border color
                    border_color = 'blue'
                    if reveal_ground_truth:
                        is_correct = cand.get("true_class") == cand["pred_class"]
                        border_color = 'green' if is_correct else 'red'
                    
                    # Label below image instead of above
                    ax.set_xlabel(
                        f"#{cand['display_idx']} ({cand['confidence']:.0%})",
                        fontsize=9, fontweight='bold'
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(3)
                else:
                    ax.set_visible(False)
        
        plt.suptitle(
            "CANDIDATES FOR REVIEW\n"
            "Green = Reference | Blue = Candidate to verify",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()
        
        # Print instructions with decision mapping
        print("BATCH VERIFICATION INSTRUCTIONS")
        print("Review the candidates above (grouped by predicted class).")
        print("Then call: batch.set_decisions([1, 0, 1, ...])")
        print(f"  Need {len(self.candidates)} decisions (indices #0 to #{len(self.candidates)-1})")


    def display_class_support(self, class_id: int, n_examples: int = 5):
        """Display support examples for a specific class (for comparison)."""
        support_indices = self.experiment.support_indices.get(class_id, [])
        
        if len(support_indices) == 0:
            print(f"No support examples for class {class_id}")
            return
        
        n_show = min(n_examples, len(support_indices))
        support_ids = [int(i) for i in support_indices[:n_show]]
        images = self.experiment.ds_train["images"][support_ids].numpy(aslist=True)
        
        fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
        if n_show == 1:
            axes = [axes]
        
        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].set_title(f"Support #{i+1}\nClass {class_id}", fontsize=11)
            axes[i].axis("off")
        
        plt.suptitle(f"Support Examples for Class {class_id}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def set_decisions(self, decisions: List[int]):
        """
        Set decisions for all candidates.
        
        Args:
            decisions: List of 1s and 0s (1=accept, 0=reject)
        """
        if len(decisions) != len(self.candidates):
            print(f"Error: Expected {len(self.candidates)} decisions, got {len(decisions)}")
            return
        
        self.decisions = decisions
        n_accept = sum(decisions)
        n_reject = len(decisions) - n_accept
        print(f"Decisions set: {n_accept} accepted, {n_reject} rejected")
    
    def apply_decisions(self) -> Dict:
        """Apply decisions, add accepted samples, and evaluate."""
        if self.decisions is None or len(self.decisions) == 0:
            print("No decisions set. Call set_decisions() first.")
            return {"added": 0}

        results_before = self.experiment.evaluate_on_val()

        # Group accepted samples by predicted class
        samples_by_class = defaultdict(list)
        added_samples = []
        
        for cand, decision in zip(self.candidates, self.decisions):
            if decision == 1:
                pred_class = int(cand["pred_class"])
                idx = int(cand["idx"])
                samples_by_class[pred_class].append(idx)
                added_samples.append({
                    "idx": idx,
                    "pred_class": pred_class,
                    "confidence": float(cand["confidence"]),
                    "true_class": cand.get("true_class"),
                    "correct": cand.get("true_class") == cand.get("pred_class"),
                })

        # Add samples to support set (grouped by class)
        samples_added = 0
        for class_id, indices in samples_by_class.items():
            self.experiment.add_to_support(indices, class_id)
            samples_added += len(indices)

        results_after = self.experiment.evaluate_on_val()

        # Log iteration to experiment history
        self.experiment.iteration += 1
        iteration_record = {
            'iteration': self.experiment.iteration,
            'samples_added': samples_added,
            'support_size': self.experiment.get_support_count(),
            'pool_size': self.experiment.get_pool_count(),
            'accuracy_before': results_before['accuracy'],
            'accuracy_after': results_after['accuracy'],
            'precision_after': results_after['precision'],
            'recall_after': results_after['recall'],
            'f1_after': results_after['f1'],
            'threshold_used': getattr(self, 'last_confidence_threshold', 0.0),
            'added_samples': added_samples,
        }
        self.experiment.history.append(iteration_record)
        
        # Print comprehensive metrics
        print(f"\n{'='*60}")
        print(f"ITERATION {self.experiment.iteration} COMPLETE")
        print(f"{'='*60}")
        print(f"  Samples added:  {samples_added}")
        print(f"  Support size:   {self.experiment.get_support_count()}")
        print(f"  Pool remaining: {self.experiment.get_pool_count()}")
        print(f"")
        print(f"  METRICS (before → after):")
        print(f"    Accuracy:  {results_before['accuracy']*100:.2f}% → {results_after['accuracy']*100:.2f}%")
        print(f"    Precision: {results_before['precision']*100:.2f}% → {results_after['precision']*100:.2f}%")
        print(f"    Recall:    {results_before['recall']*100:.2f}% → {results_after['recall']*100:.2f}%")
        print(f"    F1 Score:  {results_before['f1']*100:.2f}% → {results_after['f1']*100:.2f}%")

        return {
            "added": samples_added,
            "accuracy_before": results_before["accuracy"],
            "accuracy_after": results_after["accuracy"],
            "precision_after": results_after["precision"],
            "recall_after": results_after["recall"],
            "f1_after": results_after["f1"],
        }

# ============================================
# HELPER: Display candidates by class
# ============================================

def show_candidates_for_class(
    experiment: FewShotExperiment,
    class_id: int,
    confidence_threshold: float = 0.3,
    n_support: int = 3,
    n_candidates: int = 5
):
    """
    Display candidates for a specific class alongside support examples.
    Useful for focused verification on a single class.
    """
    # Get high confidence predictions
    high_conf = experiment.get_high_confidence_predictions(
        threshold=confidence_threshold,
        max_per_class=n_candidates
    )
    
    candidates = high_conf.get(class_id, [])
    if not candidates:
        print(f"No candidates found for class {class_id} at threshold {confidence_threshold}")
        return []
    
    # Get support examples
    support_indices = experiment.support_indices.get(class_id, [])
    n_support_show = min(n_support, len(support_indices))
    
    print(f"\\n{'='*60}")
    print(f"CLASS {class_id} VERIFICATION")
    print(f"{'='*60}")
    print(f"Support set size: {len(support_indices)}")
    print(f"Candidates found: {len(candidates)}")
    
    # Display
    n_cols = n_support_show + len(candidates)
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    
    # Support examples
    if n_support_show > 0:
        support_ids = [int(i) for i in support_indices[:n_support_show]]
        support_images = experiment.ds_train["images"][support_ids].numpy(aslist=True)
        support_images = [
            experiment.extractor._apply_preprocessing(img, experiment.ds_train, idx)
            for img, idx in zip(support_images, support_ids)
        ]
        
        for i, img in enumerate(support_images):
            axes[i].imshow(img)
            axes[i].set_title(f"SUPPORT\\nClass {class_id}", fontsize=11, color='green', fontweight='bold')
            axes[i].axis("off")
            for spine in axes[i].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('green')
                spine.set_linewidth(4)
    
    # Candidates
    for j, cand in enumerate(candidates):
        ax = axes[n_support_show + j]
        idx_int = int(cand["idx"])
        img = experiment.ds_train["images"][idx_int].numpy()
        img = experiment.extractor._apply_preprocessing(img, experiment.ds_train, idx_int)
        ax.imshow(img)
        
        is_correct = cand.get("true_class") == cand["pred_class"]
        color = 'blue' if is_correct else 'red'
        
        ax.set_title(
            f"CANDIDATE #{j}\\nConf: {cand['confidence']:.1%}\\n{'✓' if is_correct else '✗'}",
            fontsize=10, color=color, fontweight='bold'
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
    
    plt.suptitle(f"Class {class_id}: Support Examples vs Candidates", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return candidates



# ============================================
# HYBRID VERIFICATION: Spot-Check Added Samples
# ============================================
# This function displays a random sample of recently added pseudo-labels
# alongside their class reference images, so you can estimate the error rate.

def spot_check_pseudo_labels(
    experiment: FewShotExperiment,
    n_samples: int = 10,
    seed: int = None
) -> Dict:
    """
    Display a random sample of recently added pseudo-labels for visual verification.
    
    Returns a dict with the samples to review and their ground truth.
    """
    # Get the last iteration's added samples
    if not experiment.history:
        print("No pseudo-labeling iterations have been run yet.")
        return {}
    
    # Find all samples added across all iterations
    all_added = []
    for h in experiment.history:
        added_samples = h.get('added_samples', [])
        all_added.extend(added_samples)
    
    if not all_added:
        print("No samples were added during pseudo-labeling.")
        return {}
    
    # Random sample
    if seed is not None:
        np.random.seed(seed)
    
    n_to_check = min(n_samples, len(all_added))
    sample_indices = np.random.choice(len(all_added), size=n_to_check, replace=False)
    samples_to_check = [all_added[i] for i in sample_indices]
    
    print(f"{'='*70}")
    print(f"Spot check: {n_to_check} Random Pseudo-Labels (out of {len(all_added)} total)")
    print(f"{'='*70}")
    print("Compare each candidate (blue) with the reference image (green).")
    print("The ground truth (green/red) is revealed below each candidate.\n")
    
    # Display in a grid: 2 columns per sample (reference + candidate)
    n_cols = min(5, n_to_check)  # Max 5 pairs per row
    n_rows = (n_to_check + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    
    correct_count = 0
    
    for i, sample in enumerate(samples_to_check):
        row = i // n_cols
        col = (i % n_cols) * 2  # Each sample takes 2 columns
        
        pred_class = sample['pred_class']
        true_class = sample.get('true_class', pred_class)
        is_correct = (pred_class == true_class)
        if is_correct:
            correct_count += 1
        
        # Reference image (first support example for this class)
        ax_ref = axes[row, col]
        support_indices = experiment.support_indices.get(pred_class, [])
        if len(support_indices) > 0:
            ref_idx = int(support_indices[0])
            ref_img = experiment.ds_train["images"][ref_idx].numpy()
            ref_img = experiment.extractor._apply_preprocessing(ref_img, experiment.ds_train, ref_idx)
            ax_ref.imshow(ref_img)
        ax_ref.set_title(f"REF: Class {pred_class}", fontsize=9, color='green', fontweight='bold')
        ax_ref.axis('off')
        for spine in ax_ref.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('green')
            spine.set_linewidth(3)
        
        # Candidate image
        ax_cand = axes[row, col + 1]
        cand_idx = int(sample['idx'])
        cand_img = experiment.ds_train["images"][cand_idx].numpy()
        cand_img = experiment.extractor._apply_preprocessing(cand_img, experiment.ds_train, cand_idx)
        ax_cand.imshow(cand_img)
        
        # Color based on correctness
        border_color = 'green' if is_correct else 'red'
        symbol = '✓' if is_correct else '✗'
        ax_cand.set_title(
            f"{symbol} Conf: {sample['confidence']:.0%}\nTrue: {true_class}",
            fontsize=9, 
            color=border_color, 
            fontweight='bold'
        )
        ax_cand.axis('off')
        for spine in ax_cand.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
    
    # Hide unused axes
    for i in range(n_to_check, n_rows * n_cols):
        row = i // n_cols
        col = (i % n_cols) * 2
        axes[row, col].set_visible(False)
        axes[row, col + 1].set_visible(False)
    
    plt.suptitle("Spot-Check: Reference (green) vs Added Sample (green=correct, red=wrong)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    error_rate = 1 - (correct_count / n_to_check)
    print(f"Spot-Check Results")
    print(f"  Samples checked:  {n_to_check}")
    print(f"  Correct:          {correct_count} ({100*correct_count/n_to_check:.1f}%)")
    print(f"  Incorrect:        {n_to_check - correct_count} ({100*error_rate:.1f}%)")
    print(f"")
    print(f" Estimated noise rate: {100*error_rate:.1f}%")
    print(f"")
    
    if error_rate > 0.30:
        print(f"  High noise detected")
    elif error_rate > 0.15:
        print(f"  Moderate noise")
    else:
        print(f"  Low noise rate")
        
    return {
        'n_checked': n_to_check,
        'n_correct': correct_count,
        'error_rate': error_rate,
        'samples': samples_to_check
    }



# Efficientnet fine tuning helpers

class DeepLakeEffNetDataset(Dataset):
    """DeepLake dataset wrapper for EfficientNet-B4 training/eval.
    
    Supports both torchvision transforms and Albumentations pipelines.
    When custom_aug_pipeline is provided, it takes precedence over train_aug.
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
        custom_aug_pipeline: Optional[Any] = None,
    ):
        self.ds = ds
        self.indices = np.asarray(indices, dtype=int)
        self.y = np.asarray(y, dtype=int)
        self.preprocess_mode = str(preprocess_mode)
        self.bbox_padding_ratio = float(bbox_padding_ratio)
        self.custom_aug_pipeline = custom_aug_pipeline
        self.use_albumentations = custom_aug_pipeline is not None

        if self.preprocess_mode not in {"native", "bbox_crop"}:
            raise ValueError("preprocess_mode must be 'native' or 'bbox_crop'")

        base_tf = weights.transforms()
        if not self.use_albumentations:
            # Use torchvision transforms
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
        else:
            # For Albumentations, we still need base transform for final sizing
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
        
        if self.use_albumentations:
            # Apply Albumentations pipeline (includes normalization and ToTensorV2)
            augmented = self.custom_aug_pipeline(image=img)
            x = augmented["image"]
        else:
            # Apply torchvision transforms
            x = self.tf(Image.fromarray(img))
        return x, int(self.y[i])


def build_efficientnet_b4(num_classes: int, classifier_dropout: float = 0.3) -> Tuple[nn.Module, object]:
    """Build EfficientNet-B4 with custom classifier head.
    
    Args:
        num_classes: Number of output classes
        classifier_dropout: Dropout probability before final layer (reduces overfitting)
    """
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[-1].in_features
    # Replace classifier with dropout + linear for better regularization
    model.classifier = nn.Sequential(
        nn.Dropout(p=classifier_dropout, inplace=True),
        nn.Linear(in_features, int(num_classes)),
    )
    return model, weights


def freeze_backbone_effnet(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def _set_bn_eval(m: nn.Module) -> None:
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


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


@torch.no_grad()
def evaluate_with_preds(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    n = 0
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
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    mets = _macro_metrics(y_true, y_pred)
    mets["loss"] = total_loss / max(1, n)
    return mets, y_true, y_pred


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    amp: bool,
    grad_clip_norm: float = 1.0,
    desc: str = "Training",
    scheduler: Optional[Any] = None,
):
    """Train for one epoch with optional LR scheduler (per-batch stepping)."""
    model.train()
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None
    total_loss = 0.0
    n = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
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
        
        # Step the LR scheduler per batch (if provided)
        if scheduler is not None:
            scheduler.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{total_loss/n:.4f}"})

    pbar.close()
    mets = _macro_metrics(y_true, y_pred)
    mets["loss"] = total_loss / max(1, n)
    return mets


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, desc: str = "Evaluating"):
    model.eval()
    total_loss = 0.0
    n = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    
    pbar.close()
    mets = _macro_metrics(y_true, y_pred)
    mets["loss"] = total_loss / max(1, n)
    return mets


@dataclass
class TrainConfig:
    run_name: str = "effnetb4_support"
    preprocess_mode: str = "native"  # native | bbox_crop
    bbox_padding_ratio: float = 0.15
    train_aug: bool = True
    custom_aug_pipeline: Optional[Any] = None  # Albumentations Compose pipeline (overrides train_aug)

    batch_size: int = 32

    head_epochs: int = 1
    finetune_epochs: int = 3

    lr_head: float = 3e-3
    lr_backbone: float = 3e-5
    lr_head_finetune: float = 3e-4

    weight_decay: float = 1e-4
    label_smoothing: float = 0.1  # Reduces overconfidence, helps generalization

    use_weighted_sampler: bool = True
    amp: bool = True
    grad_clip_norm: float = 1.0

    freeze_bn_in_head: bool = True
    freeze_bn_in_finetune: bool = True

    # Regularization additions for few-shot learning
    classifier_dropout: float = 0.3  # Dropout before final classifier
    early_stopping_patience: int = 3  # Stop if no improvement for N epochs
    use_lr_scheduler: bool = True  # Cosine annealing LR scheduler
    warmup_epochs: int = 1  # LR warmup epochs

    seed: int = 42
    device: str = "auto"
    out_dir: str = "runs"

def train_two_stage_effnetb4(
    cfg: TrainConfig,
    ds_train,
    train_label_index: Dict[int, np.ndarray],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    """Two-stage training on the provided label_index (expanded support set)."""
    import pandas as pd

    seed_everything(cfg.seed)
    device = torch.device(get_device() if cfg.device == "auto" else cfg.device)

    class_ids = sorted(int(k) for k in train_label_index.keys())
    class_id_to_idx = {cid: i for i, cid in enumerate(class_ids)}

    # Simple stratified split within the support label_index
    rng = np.random.default_rng(cfg.seed)
    train_idx, train_y = [], []
    val_idx, val_y = [], []
    test_idx, test_y = [], []

    for cid in class_ids:
        idxs = np.asarray(train_label_index[cid], dtype=int).copy()
        if idxs.size == 0:
            continue
        rng.shuffle(idxs)

        n = len(idxs)
        n_val = max(1, int(round(val_frac * n))) if n >= 3 else (1 if n == 2 else 0)
        n_test = max(1, int(round(test_frac * n))) if n >= 4 else (0 if n <= 3 else 1)
        n_train = max(1, n - n_val - n_test)

        tr = idxs[:n_train]
        va = idxs[n_train : n_train + n_val]
        te = idxs[n_train + n_val : n_train + n_val + n_test]

        y = class_id_to_idx[cid]
        train_idx.extend(tr.tolist())
        train_y.extend([y] * len(tr))
        val_idx.extend(va.tolist())
        val_y.extend([y] * len(va))
        test_idx.extend(te.tolist())
        test_y.extend([y] * len(te))

    model, weights = build_efficientnet_b4(
        num_classes=len(class_ids),
        classifier_dropout=cfg.classifier_dropout,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    train_ds = DeepLakeEffNetDataset(
        ds_train,
        np.asarray(train_idx, dtype=int),
        np.asarray(train_y, dtype=int),
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        train_aug=cfg.train_aug,
        custom_aug_pipeline=cfg.custom_aug_pipeline,
    )
    val_ds = DeepLakeEffNetDataset(
        ds_train,
        np.asarray(val_idx, dtype=int),
        np.asarray(val_y, dtype=int),
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        train_aug=False,
        custom_aug_pipeline=None,  # No augmentation for validation
    )
    test_ds = DeepLakeEffNetDataset(
        ds_train,
        np.asarray(test_idx, dtype=int),
        np.asarray(test_y, dtype=int),
        weights=weights,
        preprocess_mode=cfg.preprocess_mode,
        bbox_padding_ratio=cfg.bbox_padding_ratio,
        train_aug=False,
        custom_aug_pipeline=None,  # No augmentation for test
    )

    sampler = _make_weighted_sampler(np.asarray(train_y, dtype=int)) if cfg.use_weighted_sampler else None
    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True, sampler=sampler)
    val_loader = _make_loader(val_ds, cfg.batch_size, shuffle=False)
    test_loader = _make_loader(test_ds, cfg.batch_size, shuffle=False)

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []

    def run_stage(stage: str, epochs: int, optimizer: torch.optim.Optimizer, freeze_bn: bool):
        nonlocal history
        best_f1 = -1.0
        best_state = None
        epochs_without_improvement = 0
        
        # Skip stage if epochs == 0
        if epochs <= 0:
            print(f"\n[Skipping {stage.upper()} stage (epochs=0)]")
            return best_f1
        
        # Setup LR scheduler if enabled
        scheduler = None
        if cfg.use_lr_scheduler and epochs > 1:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
            # Warmup + Cosine annealing
            warmup_epochs = min(cfg.warmup_epochs, epochs - 1)
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, 
                    total_iters=warmup_epochs * len(train_loader)
                )
                cosine_scheduler = CosineAnnealingWarmRestarts(
                    optimizer, T_0=(epochs - warmup_epochs) * len(train_loader), T_mult=1
                )
                scheduler = SequentialLR(
                    optimizer, 
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs * len(train_loader)]
                )
            else:
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer, T_0=epochs * len(train_loader), T_mult=1
                )
        
        print(f"\n{'='*60}")
        print(f"STAGE: {stage.upper()} ({epochs} epochs)")
        if scheduler:
            print(f"  LR Scheduler: Warmup({cfg.warmup_epochs}ep) + CosineAnnealing")
        print(f"  Early Stopping Patience: {cfg.early_stopping_patience}")
        print(f"{'='*60}")

        for ep in range(1, int(epochs) + 1):
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
                desc=f"{stage} Epoch {ep}/{epochs} [Train]",
                scheduler=scheduler,  # Pass scheduler for per-batch updates
            )
            va = evaluate(model, val_loader, device=device, criterion=criterion, desc=f"{stage} Epoch {ep}/{epochs} [Val]")
            
            elapsed = time.time() - t0
            
            # Check for improvement
            improved = va["f1"] > best_f1
            improvement_marker = "↑" if improved else ""
            
            # Get current LR for logging
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {ep}/{epochs} | "
                  f"Train Loss: {tr['loss']:.4f}, Acc: {tr['acc']*100:.1f}% | "
                  f"Val Loss: {va['loss']:.4f}, Acc: {va['acc']*100:.1f}%, F1: {va['f1']*100:.1f}% {improvement_marker} | "
                  f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

            history.append(
                {
                    "stage": stage,
                    "epoch": ep,
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
                    "lr": current_lr,
                    "seconds": elapsed,
                }
            )

            if improved:
                best_f1 = va["f1"]
                best_state = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "class_ids": class_ids}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            # Early stopping check
            if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"  ⚠️ Early stopping triggered after {cfg.early_stopping_patience} epochs without improvement")
                break

        if best_state is not None:
            model.load_state_dict(best_state["model_state"], strict=True)
            print(f"  ✓ Restored best model (Val F1: {best_f1*100:.2f}%)")
        return best_f1

    # Stage 1: head-only
    freeze_backbone_effnet(model)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    run_stage("head", cfg.head_epochs, opt1, cfg.freeze_bn_in_head)

    # Stage 2: fine-tune
    unfreeze_all(model)
    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head_finetune},
        ],
        weight_decay=cfg.weight_decay,
    )
    run_stage("finetune", cfg.finetune_epochs, opt2, cfg.freeze_bn_in_finetune)

    test_m, y_true, y_pred = evaluate_with_preds(model, test_loader, device=device, criterion=criterion)

    hist_df = pd.DataFrame(history)
    summary = {
        "test": test_m,
        "val_best_f1": float(hist_df["val_f1"].max()) if len(hist_df) else float("nan"),
        "n_classes": len(class_ids),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "class_ids": class_ids,
        "out_dir": str(out_dir),
    }

    return model, hist_df, summary, out_dir

def plot_training_curves(hist_df) -> None:
    import matplotlib.pyplot as plt

    if hist_df is None or len(hist_df) == 0:
        print("No training history to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist_df.index + 1, hist_df["train_loss"], label="train_loss")
    axes[0].plot(hist_df.index + 1, hist_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("step")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(hist_df.index + 1, hist_df["train_acc"], label="train_acc")
    axes[1].plot(hist_df.index + 1, hist_df["val_acc"], label="val_acc")
    axes[1].plot(hist_df.index + 1, hist_df["train_f1"], label="train_f1")
    axes[1].plot(hist_df.index + 1, hist_df["val_f1"], label="val_f1")
    axes[1].set_title("Accuracy/F1 (macro)")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    plt.show()

def build_label_index_from_support(exp: FewShotExperiment) -> dict:
    """Return {class_id: np.ndarray(indices)} for the current expanded support set."""
    return {
        int(cid): np.asarray(idxs, dtype=int)
        for cid, idxs in exp.support_indices.items()
        if len(idxs) > 0
    }

def _map_class_ids_to_train_y(class_ids, y_class_id: np.ndarray) -> np.ndarray:
    class_id_to_idx = {int(cid): i for i, cid in enumerate(class_ids)}
    return np.asarray([class_id_to_idx[int(c)] for c in y_class_id], dtype=int)

def evaluate_effnet_on_indices(
    model,
    ds,
    indices: np.ndarray,
    y_class_id: np.ndarray,
    class_ids,
    preprocess_mode: str,
    bbox_padding_ratio: float = 0.15,
    batch_size: int = 32,
):
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    y = _map_class_ids_to_train_y(class_ids, y_class_id)
    ds_eval = DeepLakeEffNetDataset(
        ds,
        np.asarray(indices, dtype=int),
        y,
        weights=weights,
        preprocess_mode=preprocess_mode,
        bbox_padding_ratio=bbox_padding_ratio,
        train_aug=False,
    )
    loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device(get_device())
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    mets, _, _ = evaluate_with_preds(model, loader, device=device, criterion=criterion)
    return mets
    criterion = torch.nn.CrossEntropyLoss()

class _PoolInferenceDataset(Dataset):
    def __init__(self, ds, indices: np.ndarray, preprocess_mode: str, bbox_padding_ratio: float):
        self.ds = ds
        self.indices = np.asarray(indices, dtype=int)
        self.preprocess_mode = str(preprocess_mode)
        self.bbox_padding_ratio = float(bbox_padding_ratio)
        self.weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        self.tf = self.weights.transforms()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        sample = self.ds[idx]
        img = sample["images"].numpy()
        if self.preprocess_mode == "bbox_crop":
            box = sample["boxes"].numpy()
            img = apply_bbox_crop_optimized(img, box, padding_ratio=self.bbox_padding_ratio)
        x = self.tf(Image.fromarray(img))
        return x, idx

@torch.no_grad()
def pseudo_label_pool_with_effnet(
    exp: FewShotExperiment,
    model,
    *,
    ds,
    class_ids,
    confidence_threshold: float = 0.95,
    max_per_class: int = 10,
    batch_size: int = 64,
    preprocess_mode: str = "bbox_crop",
    bbox_padding_ratio: float = 0.15,
    use_true_labels: bool = True,
):
    """Pseudo-label the remaining pool with a fine-tuned EfficientNet and add to support."""
    pool_flat = flatten_indices(exp.pool_indices)
    if len(pool_flat) == 0:
        print("Pool is empty.")
        return {"n_candidates": 0, "n_added": 0, "n_correct": 0, "n_wrong": 0}

    ds_inf = _PoolInferenceDataset(
        ds,
        pool_flat,
        preprocess_mode=preprocess_mode,
        bbox_padding_ratio=bbox_padding_ratio,
    )
    loader = DataLoader(ds_inf, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device(get_device())
    model = model.to(device)
    model.eval()

    class_ids = [int(c) for c in class_ids]

    # class_id -> list[(idx, conf, true_class_id)]
    by_class: Dict[int, List[Tuple[int, float, int]]] = {}

    for x, idxs in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        conf = conf.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        idxs = idxs.detach().cpu().numpy()

        for idx, p, c in zip(idxs, pred, conf):
            c = float(c)
            if c < confidence_threshold:
                continue
            pred_class_id = int(class_ids[int(p)])
            true_class_id = int(ds["labels"][int(idx)].numpy().squeeze())
            by_class.setdefault(pred_class_id, []).append((int(idx), c, true_class_id))

    # cap per class by confidence
    selected: Dict[int, List[Tuple[int, float, int]]] = {}
    for class_id, items in by_class.items():
        items.sort(key=lambda t: t[1], reverse=True)
        selected[class_id] = items[: int(max_per_class)]

    n_candidates = sum(len(v) for v in selected.values())
    n_added = 0
    n_correct = 0
    n_wrong = 0

    for class_id, items in selected.items():
        idxs = []
        for idx, conf, true_class_id in items:
            if use_true_labels and class_id != true_class_id:
                n_wrong += 1
                continue
            idxs.append(idx)
            n_added += 1
            if class_id == true_class_id:
                n_correct += 1
            else:
                n_wrong += 1
        if idxs:
            exp.add_to_support(idxs, class_id)

    print(f"Candidates (>= {confidence_threshold:.0%}): {n_candidates}")
    print(f"Added: {n_added} | Correct: {n_correct} | Wrong: {n_wrong}")
    print(f"New support size: {exp.get_support_count()} | Pool remaining: {exp.get_pool_count()}")
    return {"n_candidates": n_candidates, "n_added": n_added, "n_correct": n_correct, "n_wrong": n_wrong}



