from __future__ import annotations

import numpy as np
import numpy.typing as npt
import cv2
import random
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision import transforms

from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value to use for all random number generators.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device (CUDA → MPS → CPU).

    Automatically detects and returns the optimal compute device for PyTorch
    operations in order of preference: CUDA GPU, Apple MPS, or CPU.

    Returns:
        torch.device: The best available device for tensor operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_bbox_from_box_array(
    box: np.ndarray, img_h: int, img_w: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convert bounding box to pixel-space (x1, y1, x2, y2) clipped to image bounds.

    Automatically detects and handles multiple bounding box formats:
    normalized xyxy (0-1 range), xywh format, and xyxy format.

    Args:
        box: Bounding box array in any supported format (normalized xyxy, xywh,
            or xyxy). Shape should be (4,) or broadcastable to (4,).
        img_h: Image height in pixels. Must be positive.
        img_w: Image width in pixels. Must be positive.

    Returns:
        Tuple of (x1, y1, x2, y2) representing the bounding box in pixel
        coordinates, clipped to image bounds. Returns None if the box is
        invalid (wrong shape) or results in a degenerate region (zero area).

    Raises:
        None: This function does not raise exceptions; returns None on invalid input.
    """
    box = np.asarray(box, dtype=float).squeeze()
    if box.shape[-1] != 4:
        return None

    x1, y1, x2, y2 = box
    h, w = img_h, img_w
    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
    else:
        width, height = x2, y2
        if (
            width > 0
            and height > 0
            and x1 + width <= w + 1e-3
            and y1 + height <= h + 1e-3
        ):
            x2 = x1 + width
            y2 = y1 + height

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def resolve_bbox_xywh_or_xyxy(
    ds: Any, idx: int
) -> Optional[Tuple[float, float, float, float]]:
    """Load image dimensions and resolve bounding box format.

    Legacy wrapper that loads the image to get dimensions, then delegates
    to resolve_bbox_from_box_array for format detection and conversion.

    .. deprecated::
        Prefer resolve_bbox_from_box_array() when image is already loaded
        to avoid redundant data loading.

    Args:
        ds: DeepLake dataset containing 'images' and 'boxes' tensors.
        idx: Index of the sample in the dataset. Must be a valid index.

    Returns:
        Tuple of (x1, y1, x2, y2) representing the bounding box in pixel
        coordinates. Returns None if the bounding box is invalid.

    Raises:
        IndexError: If idx is out of bounds for the dataset.
        KeyError: If dataset doesn't contain 'images' or 'boxes' tensors.
    """
    img = ds["images"][idx].numpy()
    h, w = img.shape[:2]
    box = ds["boxes"][idx].numpy()
    return resolve_bbox_from_box_array(box, h, w)


def apply_bbox_crop_optimized(
    img: np.ndarray, box: np.ndarray, padding_ratio: float = 0.15
) -> np.ndarray:
    """Crop image to bounding box with padding using pre-loaded data.

    Extracts the region defined by the bounding box with additional padding
    around the edges. Uses reflection padding when the padded region extends
    beyond image boundaries.

    Args:
        img: Input image as numpy array with shape (H, W, C) or (H, W).
            Must be a valid image array.
        box: Bounding box array in any supported format (normalized xyxy,
            xywh, or xyxy).
        padding_ratio: Fraction of box dimensions to add as padding on each
            side. Default is 0.15 (15% padding). Must be non-negative.

    Returns:
        Cropped image region with padding applied. If the bounding box is
        invalid, returns the original image unchanged.

    Raises:
        None: This function does not raise exceptions; returns original
            image on invalid bounding box.
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

    # Calculate how much padding on each side
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)

    # Clip crop region to valid image bounds
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x2)
    crop_y2 = min(h, crop_y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return img
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = cv2.copyMakeBorder(
            cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )

    return cropped


def apply_bbox_crop(
    img: np.ndarray, ds: Any, idx: int, padding_ratio: float = 0.15
) -> np.ndarray:
    """Crop image to bounding box with padding using dataset lookup.

    Legacy wrapper that loads the bounding box from the dataset and
    delegates to apply_bbox_crop_optimized.

    .. deprecated::
        Use apply_bbox_crop_optimized() with pre-loaded box array for
        better performance in batch processing scenarios.

    Args:
        img: Input image as numpy array with shape (H, W, C) or (H, W).
        ds: DeepLake dataset containing 'boxes' tensor.
        idx: Index of the sample in the dataset.
        padding_ratio: Fraction of box dimensions to add as padding on each
            side. Default is 0.15 (15% padding).

    Returns:
        Cropped image region with padding applied. Returns the original
        image if box loading fails or the box is invalid.

    Raises:
        None: This function catches exceptions internally and returns the
            original image on failure.
    """
    try:
        box = ds["boxes"][idx].numpy()
    except Exception:
        return img
    return apply_bbox_crop_optimized(img, box, padding_ratio)


def build_label_index(ds: Any) -> Dict[int, npt.NDArray[np.int64]]:
    """Build a mapping from class label to array of dataset indices.

    Iterates through the entire dataset to create an index mapping each
    unique class label to all sample indices belonging to that class.
    Useful for creating stratified few-shot subsets.

    Args:
        ds: DeepLake dataset containing 'labels' tensor. The dataset must
            support iteration and have a 'labels' attribute.

    Returns:
        Dictionary mapping integer class IDs to numpy arrays of int64
        dataset indices for each class.

    Raises:
        KeyError: If dataset doesn't contain a 'labels' tensor.
        TypeError: If labels cannot be converted to integers.
    """
    label_to_idxs: Dict[int, list] = defaultdict(list)
    for i, sample in tqdm(enumerate(ds), total=len(ds), desc="Building label index"):
        label = int(sample["labels"].numpy()[0])
        label_to_idxs[label].append(int(i))
    return {k: np.array(v, dtype=np.int64) for k, v in label_to_idxs.items()}


def save_label_index(
    label_index: Dict[int, npt.NDArray[np.int64]], path: str | Path
) -> None:
    """Persist the label index to disk as a compressed numpy archive.

    Saves the label-to-indices mapping in .npz format for efficient
    loading in subsequent sessions.

    Args:
        label_index: Dictionary mapping integer class IDs to numpy arrays
            of dataset indices.
        path: File path for the output .npz file. Parent directories must
            exist. Can be a string or Path object.

    Returns:
        None.

    Raises:
        OSError: If the file cannot be written (permissions, disk space, etc.).
        TypeError: If label_index values are not numpy arrays.
    """
    np.savez_compressed(path, **{str(k): v for k, v in label_index.items()})


def load_label_index(path: str | Path) -> Dict[int, npt.NDArray[np.int64]]:
    """Load a label index from disk that was saved by save_label_index.

    Restores the label-to-indices mapping from a .npz archive.

    Args:
        path: File path to the .npz file containing the label index.
            Can be a string or Path object.

    Returns:
        Dictionary mapping integer class IDs to numpy arrays of
        dataset indices for each class.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid .npz archive.
    """
    data = np.load(path)
    return {int(k): data[k] for k in data.files}


def create_fewshot_split(
    label_index: Dict[int, npt.NDArray[np.int64]],
    n_support: int = 5,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[
    Dict[int, npt.NDArray[np.int64]],
    Dict[int, npt.NDArray[np.int64]],
    Dict[int, npt.NDArray[np.int64]],
    Dict[int, npt.NDArray[np.int64]],
]:
    """Split training dataset indices into four stratified sets.

    Creates support, pool, validation, and test splits from the training
    data while preserving class balance. The original ds_val is kept
    completely untouched as the final test set per project requirements.

    Args:
        label_index: Dictionary mapping class_id to array of dataset indices
            for that class.
        n_support: Number of support samples per class for the few-shot
            training set. Default is 5.
        val_fraction: Fraction of total class samples to reserve for
            validation. Default is 0.15 (15%).
        test_fraction: Fraction of total class samples to reserve for
            testing. Default is 0.15 (15%).
        seed: Random seed for reproducible shuffling. Default is 42.

    Returns:
        Tuple of four dictionaries (support_indices, pool_indices,
        val_indices, test_indices), each mapping class IDs to numpy
        arrays of dataset indices.

    Raises:
        None: This function handles edge cases (empty classes) gracefully.
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

        n_support_actual = min(int(n_support), n_total)
        remaining = n_total - n_support_actual

        n_val = 0
        n_test = 0
        if remaining > 0:
            if val_fraction > 0:
                n_val = max(1, int(round(n_total * val_fraction)))
            if test_fraction > 0:
                n_test = max(1, int(round(n_total * test_fraction)))
            n_val = min(n_val, remaining)
            remaining -= n_val
            n_test = min(n_test, remaining)
            remaining -= n_test

        support_indices[class_id] = indices[:n_support_actual].copy()
        val_indices[class_id] = indices[
            n_support_actual : n_support_actual + n_val
        ].copy()
        test_indices[class_id] = indices[
            n_support_actual + n_val : n_support_actual + n_val + n_test
        ].copy()
        pool_indices[class_id] = indices[n_support_actual + n_val + n_test :].copy()

    return support_indices, pool_indices, val_indices, test_indices


def flatten_indices(
    indices_dict: Dict[int, npt.NDArray[np.int64]],
) -> npt.NDArray[np.int64]:
    """Flatten a dictionary of class-to-indices mappings to a single array.

    Concatenates all index arrays from the dictionary values into a single
    1D array, useful for batch processing across all classes.

    Args:
        indices_dict: Dictionary mapping class IDs to numpy arrays of
            dataset indices.

    Returns:
        1D numpy array of int64 containing all indices from all classes
        combined. Returns empty array if input is empty.

    Raises:
        None: This function handles empty dictionaries gracefully.
    """
    all_indices = []
    for indices in indices_dict.values():
        all_indices.extend(indices)
    return np.array(all_indices, dtype=np.int64)


def get_labels_for_indices(
    ds: Any, indices: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Retrieve class labels for a set of dataset indices.

    Fetches labels from the dataset for the specified indices, handling
    the conversion from DeepLake tensor format to a flat numpy array.

    Args:
        ds: DeepLake dataset containing 'labels' tensor.
        indices: Array of dataset indices to retrieve labels for.
            All indices must be valid for the dataset.

    Returns:
        1D numpy array of int64 class labels corresponding to the
        input indices, in the same order.

    Raises:
        IndexError: If any index is out of bounds for the dataset.
        KeyError: If dataset doesn't contain a 'labels' tensor.
    """
    indices_list = [int(i) for i in indices]
    labels_np = ds["labels"][indices_list].numpy().astype(int)
    return labels_np.reshape(len(labels_np), -1)[:, 0]


class MultiBackboneFeatureExtractor:
    """Feature extractor supporting multiple pretrained backbone architectures.

    Extracts fixed-size embedding vectors from images using ImageNet-pretrained
    CNN or Vision Transformer backbones. Supports configurable preprocessing
    modes including native transforms and bounding box cropping.

    Supported backbones:
        - 'resnet50': ResNet-50 (2048-dim embeddings)
        - 'efficientnet_b4': EfficientNet-B4 (1792-dim embeddings)
        - 'vit_b_16': Vision Transformer B/16 (768-dim embeddings)

    Preprocessing modes:
        - 'native': Backbone's ImageNet transforms (optionally pad-to-square first)
        - 'bbox_crop': Bbox crop (+padding) then backbone transforms

    Attributes:
        backbone_name: Name of the backbone architecture.
        device: PyTorch device for computation.
        preprocess_mode: Current preprocessing mode.
        pad_to_square: Whether to pad images to square before transforms.
        bbox_padding_ratio: Padding ratio for bounding box crops.
        model: The backbone model with classifier removed.
        preprocess: Preprocessing transforms from backbone weights.
        embedding_dim: Dimension of output embeddings.
    """

    SUPPORTED_BACKBONES = ["resnet50", "efficientnet_b4", "vit_b_16"]
    SUPPORTED_PREPROCESS_MODES = ["native", "bbox_crop"]

    def __init__(
        self,
        backbone_name: str,
        device: torch.device,
        preprocess_mode: str = "native",
        pad_to_square: bool = True,
        bbox_padding_ratio: float = 0.15,
    ):
        """Initialize the feature extractor with specified backbone.

        Args:
            backbone_name: Name of the backbone architecture. Must be one of
                'resnet50', 'efficientnet_b4', or 'vit_b_16'.
            device: PyTorch device for model inference (e.g., torch.device('cuda')).
            preprocess_mode: Preprocessing mode. Either 'native' for standard
                ImageNet transforms or 'bbox_crop' for bounding box cropping.
            pad_to_square: Whether to pad images to square before applying
                backbone transforms. Helps preserve aspect ratio.
            bbox_padding_ratio: Fraction of box dimensions to add as padding
                when using 'bbox_crop' mode. Default is 0.15.

        Raises:
            ValueError: If backbone_name is not a supported architecture.
            ValueError: If preprocess_mode is not 'native' or 'bbox_crop'.
        """
        if backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone must be one of {self.SUPPORTED_BACKBONES}")
        if preprocess_mode not in self.SUPPORTED_PREPROCESS_MODES:
            raise ValueError(
                f"Preprocess mode must be one of {self.SUPPORTED_PREPROCESS_MODES}"
            )

        self.backbone_name = backbone_name
        self.device = device
        self.preprocess_mode = preprocess_mode
        self.pad_to_square = bool(pad_to_square)
        self.bbox_padding_ratio = float(bbox_padding_ratio)
        if backbone_name == "resnet50":
            self._init_resnet50()
        elif backbone_name == "efficientnet_b4":
            self._init_efficientnet_b4()
        elif backbone_name == "vit_b_16":
            self._init_vit_b_16()

        self.model = self.model.to(device).eval()
        pad_tag = "pad" if self.pad_to_square else "no-pad"
        print(
            f"Loaded {backbone_name} | mode={preprocess_mode} | {pad_tag} | dim={self.embedding_dim}"
        )

    def _pad_to_square(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Pad a rectangular image to square using reflection.

        Adds symmetric padding to the shorter dimension using BORDER_REFLECT_101
        to create a square image while preserving content.

        Args:
            img: Input image as numpy array with shape (H, W, C) or (H, W).

        Returns:
            Square image with reflection padding applied. If already square,
            returns the original image unchanged.

        Raises:
            None: This function handles all valid image arrays.
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
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )

    def _init_resnet50(self) -> None:
        """Initialize ResNet-50 backbone with ImageNet-pretrained weights.

        Sets up the model with the final fully connected layer replaced by
        Identity to output 2048-dimensional embeddings.

        Returns:
            None. Sets self.model, self.preprocess, and self.embedding_dim.

        Raises:
            RuntimeError: If pretrained weights cannot be downloaded.
        """
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Identity()
        self.preprocess = weights.transforms()
        self.embedding_dim = 2048

    def _init_efficientnet_b4(self) -> None:
        """Initialize EfficientNet-B4 backbone with ImageNet-pretrained weights.

        Sets up the model with the classifier replaced by Identity to output
        1792-dimensional embeddings.

        Returns:
            None. Sets self.model, self.preprocess, and self.embedding_dim.

        Raises:
            RuntimeError: If pretrained weights cannot be downloaded.
        """
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b4(weights=weights)
        self.model.classifier = nn.Identity()
        self.preprocess = weights.transforms()
        self.embedding_dim = 1792

    def _init_vit_b_16(self) -> None:
        """Initialize Vision Transformer B/16 with ImageNet-pretrained weights.

        Sets up the model with the heads replaced by Identity to output
        768-dimensional embeddings.

        Returns:
            None. Sets self.model, self.preprocess, and self.embedding_dim.

        Raises:
            RuntimeError: If pretrained weights cannot be downloaded.
        """
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = models.vit_b_16(weights=weights)
        self.model.heads = nn.Identity()
        self.preprocess = weights.transforms()
        self.embedding_dim = 768

    def _apply_preprocessing(
        self, img: np.ndarray, ds=None, idx: int = None, box: np.ndarray = None
    ) -> np.ndarray:
        """Apply preprocessing based on the configured mode.

        Handles both native and bbox_crop preprocessing modes, optionally
        padding to square before returning.

        Args:
            img: Input image as numpy array with shape (H, W, C).
            ds: Optional DeepLake dataset for bbox crop mode. Used when
                box parameter is not provided (legacy usage).
            idx: Optional sample index for bbox crop mode. Required if ds
                is provided and box is None.
            box: Optional pre-loaded bounding box array. Preferred over
                ds+idx for batch processing efficiency.

        Returns:
            Preprocessed image ready for backbone transforms. Shape may
            change if pad_to_square is enabled.

        Raises:
            None: Invalid inputs result in returning the original image.
        """
        if self.preprocess_mode == "bbox_crop":
            if box is not None:
                img = apply_bbox_crop_optimized(
                    img, box, padding_ratio=self.bbox_padding_ratio
                )
            elif ds is not None and idx is not None:
                img = apply_bbox_crop(
                    img, ds, idx, padding_ratio=self.bbox_padding_ratio
                )
        if self.pad_to_square:
            img = self._pad_to_square(img)
        return img

    @torch.no_grad()
    def extract_single(self, image: np.ndarray, ds=None, idx: int = None) -> np.ndarray:
        """Extract embedding for a single image.

        Applies preprocessing, converts to tensor, and runs through the
        backbone model to produce a fixed-size embedding vector.

        Args:
            image: Input image as numpy array with shape (H, W, C).
                Should be uint8 RGB format.
            ds: Optional DeepLake dataset for bbox crop preprocessing.
            idx: Optional sample index for bbox crop preprocessing.

        Returns:
            1D numpy array embedding vector of size self.embedding_dim.

        Raises:
            RuntimeError: If model inference fails.
        """
        image = self._apply_preprocessing(image, ds, idx)
        pil_img = Image.fromarray(image)
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        return embedding.cpu().numpy().flatten()

    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for a batch of pre-processed images.

        Stacks images into a batch tensor and runs through the backbone.
        Uses automatic mixed precision on CUDA for efficiency.

        Args:
            images: List of preprocessed images as numpy arrays. Each image
                should have shape (H, W, C) in uint8 RGB format. All images
                should be preprocessed (cropped, padded) before calling.

        Returns:
            2D numpy array of shape (batch_size, embedding_dim) containing
            one embedding per input image.

        Raises:
            RuntimeError: If model inference fails.
            ValueError: If images list is empty.
        """
        tensors = torch.stack(
            [self.preprocess(Image.fromarray(img)) for img in images]
        ).to(self.device)

        if self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                embeddings = self.model(tensors)
        else:
            embeddings = self.model(tensors)

        return embeddings.float().cpu().numpy()

    @torch.no_grad()
    def extract_from_dataset(
        self, ds, indices: np.ndarray, batch_size: int = 64, show_progress: bool = True
    ) -> np.ndarray:
        """Extract embeddings for specific dataset indices in batches.

        Loads images from the dataset, applies preprocessing, and extracts
        embeddings in batches for memory efficiency.

        Args:
            ds: DeepLake dataset containing 'images' tensor and optionally
                'boxes' tensor for bbox_crop mode.
            indices: Array of dataset indices to extract embeddings for.
                All indices must be valid for the dataset.
            batch_size: Number of images to process per batch. Larger values
                use more GPU memory but may be faster. Default is 64.
            show_progress: Whether to display a tqdm progress bar.
                Default is True.

        Returns:
            2D numpy array of shape (len(indices), embedding_dim) containing
            embeddings in the same order as input indices.

        Raises:
            IndexError: If any index is out of bounds for the dataset.
            KeyError: If dataset doesn't contain required tensors.
        """
        all_embeddings = []
        iterator = range(0, len(indices), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Extracting [{self.backbone_name}|{self.preprocess_mode}]",
            )

        for i in iterator:
            batch_indices = [int(j) for j in indices[i : i + batch_size]]
            images_np = ds["images"][batch_indices].numpy(aslist=True)
            boxes = None
            if self.preprocess_mode == "bbox_crop":
                try:
                    boxes = ds["boxes"][batch_indices].numpy(aslist=True)
                except Exception:
                    boxes = [None] * len(batch_indices)
            else:
                boxes = [None] * len(batch_indices)
            images = [
                self._apply_preprocessing(img, box=box)
                for img, box in zip(images_np, boxes)
            ]

            embeddings = self.extract_batch(images)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


def visualize_preprocessing_modes(
    ds: Any,
    indices: List[int],
    backbone_name: str = "resnet50",
    padding_ratio: float = 0.15,
    pad_to_square: bool = True,
    figsize: Tuple[int, int] = (18, 12),
) -> None:
    """Visualize preprocessing pipelines for a given backbone architecture.

    Creates a comparison grid showing original images alongside their preprocessed
    versions for both native and bounding box crop modes. Useful for understanding
    and debugging preprocessing choices.

    Args:
        ds: DeepLake dataset containing 'images', 'boxes', and 'labels' tensors.
        indices: List of dataset indices to visualize.
        backbone_name: Name of the backbone architecture. One of 'resnet50',
            'efficientnet_b4', or 'vit_b_16'.
        padding_ratio: Fraction of box dimensions to add as padding for bbox crops.
        pad_to_square: Whether to pad images to square before backbone transforms.
        figsize: Figure size as (width, height) tuple.

    Returns:
        None: Displays matplotlib figure.

    Raises:
        ValueError: If backbone_name is not one of the supported architectures.
    """
    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    elif backbone_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    elif backbone_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    else:
        raise ValueError(
            "backbone_name must be one of: resnet50, efficientnet_b4, vit_b_16"
        )

    preprocess = weights.transforms()
    mean_vals = getattr(preprocess, "mean", None) or weights.meta.get(
        "mean", [0.0, 0.0, 0.0]
    )
    std_vals = getattr(preprocess, "std", None) or weights.meta.get(
        "std", [1.0, 1.0, 1.0]
    )
    mean = torch.tensor(mean_vals).view(-1, 1, 1)
    std = torch.tensor(std_vals).view(-1, 1, 1)
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
        t_cpu = t.detach().cpu().float()
        if t_cpu.min() < -0.05 or t_cpu.max() > 1.05:
            img_t = t_cpu * std + mean
        else:
            img_t = t_cpu
        img_t = img_t.clamp(0, 1)
        return img_t.permute(1, 2, 0).numpy()

    def _preprocess_for_display(pil_img: Image.Image) -> np.ndarray:
        if display_preprocess is not None:
            out = display_preprocess(pil_img)
            if isinstance(out, torch.Tensor):
                t_cpu = out.detach().cpu().float()
                if t_cpu.max() > 1.5:
                    t_cpu = t_cpu / 255.0
                t_cpu = t_cpu.clamp(0, 1)
                return t_cpu.permute(1, 2, 0).numpy()
            arr = _to_uint8_rgb(np.asarray(out))
            return arr.astype(np.float32) / 255.0

        t = preprocess(pil_img)
        return _tensor_to_display(t)

    n_samples = len(indices)

    fig, axes = plt.subplots(n_samples, 4, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    col_titles = [
        "Original",
        f"Native ({backbone_name})",
        f"Bbox crop → native ({backbone_name})",
        "Bbox on Original",
    ]

    for row, idx in enumerate(indices):
        img = _to_uint8_rgb(ds["images"][idx].numpy())
        bbox = resolve_bbox_xywh_or_xyxy(ds, idx)
        class_id = int(ds["labels"][idx].numpy().item())
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original\n{img.shape[1]}×{img.shape[0]}", fontsize=9)
        axes[row, 0].axis("off")

        # Col 1: After backbone-native preprocessing (exactly as fed into the model)
        h, w = img.shape[:2]
        img_native_geom = _pad_to_square(img) if pad_to_square else img
        native_img = _preprocess_for_display(Image.fromarray(img_native_geom))
        axes[row, 1].imshow(native_img)
        axes[row, 1].set_title(
            f"Native\n{native_img.shape[1]}×{native_img.shape[0]}", fontsize=9
        )
        axes[row, 1].axis("off")

        # Col 2: After bbox crop + backbone-native preprocessing (exactly as fed into the model)
        img_bbox = apply_bbox_crop(img, ds, idx, padding_ratio=padding_ratio)
        img_bbox = _to_uint8_rgb(img_bbox)
        img_bbox_geom = _pad_to_square(img_bbox) if pad_to_square else img_bbox
        bbox_img = _preprocess_for_display(Image.fromarray(img_bbox_geom))
        axes[row, 2].imshow(bbox_img)
        axes[row, 2].set_title(
            f"Bbox→Native\n{bbox_img.shape[1]}×{bbox_img.shape[0]}", fontsize=9
        )
        axes[row, 2].axis("off")

        # Col 3: Original with bbox overlay
        axes[row, 3].imshow(img)
        coverage = 0.0
        if bbox is not None:
            x1, y1, x2, y2 = map(float, bbox)
            coverage = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (h * w) * 100.0
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2
            )
            axes[row, 3].add_patch(rect)
            box_w, box_h = x2 - x1, y2 - y1
            pad_x, pad_y = int(box_w * padding_ratio), int(box_h * padding_ratio)
            rect_padded = plt.Rectangle(
                (max(0, x1 - pad_x), max(0, y1 - pad_y)),
                min(w, x2 + pad_x) - max(0, x1 - pad_x),
                min(h, y2 + pad_y) - max(0, y1 - pad_y),
                fill=False,
                edgecolor="yellow",
                linewidth=1,
                linestyle="--",
            )
            axes[row, 3].add_patch(rect_padded)
        axes[row, 3].set_title(f"Bbox overlay\nCoverage: {coverage:.1f}%", fontsize=9)
        axes[row, 3].axis("off")
        axes[row, 0].set_ylabel(
            f"Class {class_id}\nIdx {idx}",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=40,
        )

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(
            title + "\n" + ax.get_title().split("\n")[-1],
            fontsize=10,
            fontweight="bold",
        )

    plt.suptitle(
        "Preprocessing Comparison: Native Center Crop vs Bounding Box Crop",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    print("\n Preprocessing Summary:")
    print(f"  • Backbone: {backbone_name}")
    print(f"  • Pad-to-square before transforms: {pad_to_square}")
    print("  • Display: de-normalized for visualization")
    print("  • Native: (pad-to-square →) weights.transforms()")
    print(
        f"  • Bbox Crop: bbox crop (+{int(padding_ratio*100)}% padding) → (pad-to-square →) weights.transforms()"
    )
    print(
        f"\n  Green box = tight bbox | Yellow dashed = bbox + {int(padding_ratio*100)}% padding"
    )


def evaluate_backbone_fewshot(
    backbone_name: str,
    preprocess_mode: str,
    ds: Any,
    val_indices: npt.NDArray[np.int64],
    val_labels: npt.NDArray[np.int64],
    support_indices: Dict[int, npt.NDArray[np.int64]],
    device: torch.device,
    batch_size: int = 64,
    max_val_samples: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a backbone and preprocessing mode for few-shot classification.

    Computes class prototypes from support set embeddings and evaluates
    classification performance on validation samples using cosine similarity.
    Supports caching of embeddings for efficient repeated evaluation.

    Args:
        backbone_name: Name of the backbone architecture. One of 'resnet50',
            'efficientnet_b4', or 'vit_b_16'.
        preprocess_mode: Preprocessing mode. Either 'native' or 'bbox_crop'.
        ds: DeepLake dataset containing 'images' and 'boxes' tensors.
        val_indices: Array of validation sample indices to evaluate.
        val_labels: Array of ground truth labels for validation samples.
            Must have same length as val_indices.
        support_indices: Dictionary mapping class IDs to arrays of support
            indices used to compute prototypes.
        device: PyTorch device for computation (e.g., torch.device('cuda')).
        batch_size: Batch size for embedding extraction. Default is 64.
        max_val_samples: Optional limit on number of validation samples.
            If specified, random subset is selected. Default is None.
        cache_dir: Optional directory for caching embeddings. If None,
            no caching is performed.

    Returns:
        Dictionary containing evaluation metrics:
            - backbone: Backbone name.
            - preprocess_mode: Preprocessing mode used.
            - config: Combined config name.
            - accuracy: Classification accuracy.
            - precision: Weighted precision.
            - recall: Weighted recall.
            - f1: Weighted F1 score.
            - embedding_dim: Dimension of embeddings.
            - time_seconds: Total evaluation time.
            - n_val_samples: Number of validation samples evaluated.
            - mean_confidence: Mean prediction confidence.
            - n_val_dropped: Samples dropped due to missing classes.

    Raises:
        ValueError: If support set is empty (no classes have samples).
        ValueError: If no validation samples remain after filtering.
    """
    PREPROCESS_IMPL_VERSION = "v3_bboxcrop_pad_to_square_then_native_transforms"
    config_name = f"{backbone_name}_{preprocess_mode}"
    print(f"\n{'='*60}")
    print(f"EVALUATING: {config_name.upper()}")
    print(f"{'='*60}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    extractor = MultiBackboneFeatureExtractor(backbone_name, device, preprocess_mode)

    # Cache setup
    cache_path_val = None
    cache_path_support = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path_val = cache_dir / f"{config_name}_val_cache.npz"
        cache_path_support = cache_dir / f"{config_name}_support_cache.npz"

    # Compute prototypes from support set
    print("\nComputing class prototypes from support set...")
    class_ids_list_all = sorted(support_indices.keys())

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
                cached_ver = (
                    cached["preprocess_impl_version"].item()
                    if "preprocess_impl_version" in cached
                    else None
                )
                cached_class_ids = (
                    cached["class_ids"] if "class_ids" in cached else None
                )
                cached_support_idx = (
                    cached["support_indices_flat"]
                    if "support_indices_flat" in cached
                    else None
                )
                cached_support_cids = (
                    cached["support_class_ids_flat"]
                    if "support_class_ids_flat" in cached
                    else None
                )
                cached_dim = (
                    int(cached["embedding_dim"]) if "embedding_dim" in cached else None
                )

                ver_match = cached_ver == PREPROCESS_IMPL_VERSION
                dim_match = cached_dim == extractor.embedding_dim
                class_ids_match = cached_class_ids is not None and np.array_equal(
                    cached_class_ids, class_ids_arr
                )
                support_match = (
                    cached_support_idx is not None
                    and cached_support_cids is not None
                    and np.array_equal(cached_support_idx, support_indices_flat)
                    and np.array_equal(cached_support_cids, support_class_ids_flat)
                )

                if ver_match and dim_match and class_ids_match and support_match:
                    prototypes = cached["prototypes"]
                    shape_ok = (
                        prototypes.shape[0] == n_classes
                        and prototypes.shape[1] == extractor.embedding_dim
                    )
                    if shape_ok:
                        support_cache_valid = True
                        print(
                            f"  Support cache valid: loaded {len(prototypes)} prototypes"
                        )
                    else:
                        print(" Support cache invalid: prototype shape mismatch")
        except Exception as e:
            print(f"  Support cache load failed: {e}")

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
            np.savez(
                cache_path_support,
                prototypes=prototypes,
                embedding_dim=extractor.embedding_dim,
                class_ids=class_ids_arr,
                support_indices_flat=support_indices_flat,
                support_class_ids_flat=support_class_ids_flat,
                preprocess_impl_version=PREPROCESS_IMPL_VERSION,
            )
            print(f"  Cached prototypes to {cache_path_support}")

    # Handle max_val_samples limit
    if max_val_samples is not None and len(val_indices) > max_val_samples:
        rng = np.random.default_rng(42)
        subset_idx = rng.choice(len(val_indices), max_val_samples, replace=False)
        current_val_indices = val_indices[subset_idx]
        current_val_labels = val_labels[subset_idx]
    else:
        current_val_indices = val_indices
        current_val_labels = val_labels

    # Ensure only score on labels that exist in the support set
    n_val_dropped = 0
    val_mask = np.isin(current_val_labels, class_ids_arr)
    if not np.all(val_mask):
        n_val_dropped = int((~val_mask).sum())
        dropped_labels = np.unique(current_val_labels[~val_mask])
        print(
            f" Dropping {n_val_dropped} val samples with labels not in support set: {dropped_labels.tolist()}"
        )
        current_val_indices = current_val_indices[val_mask]
        current_val_labels = current_val_labels[val_mask]
    if len(current_val_indices) == 0:
        raise ValueError(
            "No validation samples remain after filtering to support classes."
        )

    n_val = len(current_val_indices)

    # Extract validation embeddings
    print(f"\nExtracting validation embeddings ({n_val} samples)...")

    cache_valid = False
    if cache_path_val and cache_path_val.exists() and max_val_samples is None:
        try:
            with np.load(cache_path_val) as cached:
                cached_ver = (
                    cached["preprocess_impl_version"].item()
                    if "preprocess_impl_version" in cached
                    else None
                )
                cached_indices = cached["indices"]
                cached_embeddings = cached["embeddings"]

                indices_match = len(cached_indices) == len(
                    current_val_indices
                ) and np.array_equal(cached_indices, current_val_indices)
                dim_match = cached_embeddings.shape[1] == extractor.embedding_dim

                ver_match = cached_ver == PREPROCESS_IMPL_VERSION

                if indices_match and dim_match and ver_match:
                    print(
                        f"  ✓ Val cache valid: indices match, dim={cached_embeddings.shape[1]}"
                    )
                    val_embeddings = cached_embeddings
                    cache_valid = True
                else:
                    if not indices_match:
                        print(f"  Cache invalid: indices changed")
                    if not dim_match:
                        print(f"  Cache invalid: dim mismatch")
                    if not ver_match:
                        print(f"  Cache invalid: preprocess version mismatch")
        except Exception as e:
            print(f" Cache load failed: {e}")

    if not cache_valid:
        val_embeddings = extractor.extract_from_dataset(
            ds, current_val_indices, batch_size
        )
        if cache_path_val and max_val_samples is None:
            np.savez(
                cache_path_val,
                indices=current_val_indices,
                embeddings=val_embeddings,
                embedding_dim=extractor.embedding_dim,
                preprocess_impl_version=PREPROCESS_IMPL_VERSION,
            )
            print(f"  Cached embeddings to {cache_path_val}")

    # Classify using cosine similarity
    print("Classifying validation samples...")
    prototypes_norm = prototypes / (
        np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
    )
    val_emb_norm = val_embeddings / (
        np.linalg.norm(val_embeddings, axis=1, keepdims=True) + 1e-8
    )
    similarities = val_emb_norm @ prototypes_norm.T

    # Softmax for confidence
    logits = similarities * 10
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    pred_indices = np.argmax(similarities, axis=1)
    predictions = class_ids_arr[pred_indices]
    confidences = np.max(probs, axis=1)

    # Compute metrics
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


class FewShotExperiment:
    """Manages iterative few-shot learning experiment with pseudo-labeling.

    This class orchestrates the few-shot learning workflow including:
    - Managing support set, pool, validation, and test splits
    - Computing and caching embeddings for efficient prototype computation
    - Running pseudo-labeling iterations with configurable thresholds
    - Tracking metrics and history across iterations

    The validation/test splits are derived from training data for intermediate
    evaluation, while ds_val (original validation set) is kept untouched for
    final evaluation to prevent data leakage.

    Attributes:
        ds_train: DeepLake training dataset.
        support_indices: Dict mapping class IDs to support set indices.
        pool_indices: Dict mapping class IDs to unlabeled pool indices.
        val_indices: Dict mapping class IDs to validation indices.
        test_indices: Dict mapping class IDs to test indices.
        prototypes: Current class prototype embeddings.
        history: List of iteration statistics.
    """

    def __init__(
        self,
        ds_train: Any,
        support_indices: Dict[int, npt.NDArray[np.int64]],
        pool_indices: Dict[int, npt.NDArray[np.int64]],
        val_indices: Dict[int, npt.NDArray[np.int64]],
        test_indices: Dict[int, npt.NDArray[np.int64]],
        extractor: MultiBackboneFeatureExtractor,
        n_support: int,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        batch_size: int = 64,
        use_fp16_embeddings: bool = True,
    ) -> None:
        """Initialize the few-shot experiment.

        Sets up the experiment state including support set, pool, validation,
        and test splits. Loads or computes embeddings for the training dataset.

        Args:
            ds_train: DeepLake training dataset containing images and labels.
            support_indices: Dict mapping class IDs to initial support indices.
                These are the labeled examples for few-shot learning.
            pool_indices: Dict mapping class IDs to unlabeled pool indices.
                These samples are available for pseudo-labeling.
            val_indices: Dict mapping class IDs to validation indices.
                Used for monitoring during pseudo-labeling.
            test_indices: Dict mapping class IDs to test indices from training
                data. Used for intermediate evaluation.
            extractor: MultiBackboneFeatureExtractor instance for embedding
                computation.
            n_support: Initial number of support samples per class.
            seed: Random seed for reproducibility. Default is 42.
            cache_dir: Directory for caching embeddings. If None, uses
                'data/embedding_cache'.
            batch_size: Batch size for embedding extraction. Default is 64.
            use_fp16_embeddings: Whether to save embeddings as float16 to
                reduce disk usage by 50%. Default is True.

        Raises:
            OSError: If cache directory cannot be created.
        """
        self.ds_train = ds_train
        self.extractor = extractor
        self.n_support = n_support
        self.batch_size = batch_size
        self.use_fp16_embeddings = use_fp16_embeddings
        self.prototype_method: str = "mean"
        self.prototype_trim_k: int = 1
        self.prototype_weight_power: float = 2.0

        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/embedding_cache")

        self.backbone_name = getattr(extractor, "backbone_name", "resnet50")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.support_indices = {k: v.copy() for k, v in support_indices.items()}
        self.pool_indices = {k: v.copy() for k, v in pool_indices.items()}
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.val_flat = flatten_indices(self.val_indices)
        self.test_flat = flatten_indices(self.test_indices)
        val_labels_list = []
        for class_id, indices in self.val_indices.items():
            val_labels_list.extend([class_id] * len(indices))
        self._val_labels = np.array(val_labels_list, dtype=np.int64)

        self.history = []
        self.iteration = 0
        self.per_class_stats = {}
        self._train_embeddings = self._load_or_compute_embeddings(
            ds_train, f"{self.backbone_name}_train", len(ds_train)
        )
        self.ds_final_test = None
        self._final_test_embeddings = None
        self._final_test_labels = None
        self.prototypes, self.class_ids = self._compute_prototypes_from_cache()
        self.class_id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}

    def set_projection_head(
        self,
        projection_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        recompute: bool = True,
    ) -> None:
        """Set a learned projection head to transform embeddings.

        Configures a neural network module to project raw backbone embeddings
        into a learned space before prototype computation. Useful for improving
        embedding quality during pseudo-labeling.

        Args:
            projection_model: A PyTorch nn.Module that transforms embeddings.
                Should accept (N, input_dim) and output (N, output_dim).
                If None, removes any existing projection (uses raw embeddings).
            device: Device to run projection on. If None, uses CUDA if
                available, otherwise CPU.
            recompute: If True, immediately recompute prototypes with the
                new projection. Default is True.

        Returns:
            None.

        Raises:
            RuntimeError: If projection model cannot be moved to device.
        """
        self._projection_model = projection_model
        self._projection_device = (
            device
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )

        self._projected_train_embeddings = None
        self._projected_val_embeddings = None

        if projection_model is not None:
            projection_model.to(self._projection_device)
            projection_model.eval()
            print(f"Projection head set: {projection_model.__class__.__name__}")
            print(f"   Device: {self._projection_device}")
        else:
            print("Projection head removed, using raw embeddings")

        if recompute:
            self.prototypes, self.class_ids = self._compute_prototypes_from_cache()
            self.class_id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}

    def _apply_projection(
        self, embeddings: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Apply the projection head to embeddings if one is set.

        Transforms raw backbone embeddings through the configured projection
        model. If no projection is set, returns embeddings unchanged.

        Args:
            embeddings: Raw backbone embeddings of shape (N, D) where D is
                the backbone's embedding dimension.

        Returns:
            Projected embeddings of shape (N, D') where D' is the projection
            output dimension. Returns original embeddings if no projection
            is configured.

        Raises:
            RuntimeError: If projection model inference fails.
        """
        if not hasattr(self, "_projection_model") or self._projection_model is None:
            return embeddings

        with torch.no_grad():
            emb_tensor = (
                torch.from_numpy(embeddings).float().to(self._projection_device)
            )
            if hasattr(self._projection_model, "get_embedding"):
                projected = self._projection_model.get_embedding(emb_tensor)
            else:
                projected = self._projection_model(emb_tensor)
            return projected.cpu().numpy()

    def _get_projected_train_embeddings(self) -> npt.NDArray[np.floating[Any]]:
        """Get train embeddings with projection applied.

        Returns cached projected embeddings if available. Otherwise, applies
        projection to raw embeddings and caches the result.

        Returns:
            2D numpy array of projected training embeddings with shape
            (n_train_samples, projection_dim). If no projection is set,
            returns raw embeddings.

        Raises:
            RuntimeError: If projection model inference fails.
        """
        if (
            not hasattr(self, "_projected_train_embeddings")
            or self._projected_train_embeddings is None
        ):
            self._projected_train_embeddings = self._apply_projection(
                self._train_embeddings
            )
        return self._projected_train_embeddings

    def _load_or_compute_embeddings(
        self, dataset: Any, name: str, n_samples: int
    ) -> npt.NDArray[np.floating[Any]]:
        """Load embeddings from cache or compute and save them.

        Checks for cached embeddings on disk. If found, loads and returns
        them. Otherwise, extracts embeddings from the dataset and saves
        them to cache for future use.

        Args:
            dataset: DeepLake dataset to extract embeddings from.
            name: Name identifier for the cache file. Used to construct
                the filename as '{name}_embeddings.npy'.
            n_samples: Total number of samples in the dataset. Used for
                progress bar and validation.

        Returns:
            2D numpy array of embeddings with shape (n_samples, embedding_dim).
            Always returned as float32 for computation.

        Raises:
            OSError: If cache file cannot be read or written.
        """
        cache_path = self.cache_dir / f"{name}_embeddings.npy"

        if cache_path.exists():
            print(f"Loading cached {name} embeddings...")
            embeddings = np.load(cache_path)
            if embeddings.dtype == np.float16:
                embeddings = embeddings.astype(np.float32)
            print(f"   Loaded {len(embeddings)} embeddings from cache")
            return embeddings

        print(f"Computing {name} embeddings (one-time, will be cached).")
        embeddings = self._extract_with_progress(
            dataset, n_samples, batch_size=self.batch_size
        )
        if self.use_fp16_embeddings:
            np.save(cache_path, embeddings.astype(np.float16))
            print(f"Saved as float16 to {cache_path}")
        else:
            np.save(cache_path, embeddings)

        return embeddings

    def _extract_with_progress(
        self, dataset: Any, n_samples: int, batch_size: int = 64
    ) -> npt.NDArray[np.floating[Any]]:
        """Extract embeddings with batched reads and progress bar.

        Iterates through the dataset in batches, extracting embeddings
        while showing progress. Used internally by _load_or_compute_embeddings.

        Args:
            dataset: DeepLake dataset to extract embeddings from.
            n_samples: Total number of samples to process.
            batch_size: Number of samples per batch. Default is 64.

        Returns:
            2D numpy array of embeddings with shape (n_samples, embedding_dim).

        Raises:
            RuntimeError: If model inference fails.
            IndexError: If dataset access fails.
        """
        all_embeddings = []
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(
                range(0, n_samples, batch_size),
                total=n_batches,
                desc="Extracting features",
            ):
                end_idx = min(i + batch_size, n_samples)
                batch_indices = list(range(i, end_idx))
                images_np = dataset["images"][batch_indices].numpy(aslist=True)
                images = [
                    self.extractor._apply_preprocessing(img, dataset, idx)
                    for img, idx in zip(images_np, batch_indices)
                ]
                embeddings = self.extractor.extract_batch(images)
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def _compute_prototypes_from_cache(
        self,
    ) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.int64]]:
        """Compute class prototypes from cached embeddings.

        Computes the prototype embedding for each class by aggregating
        support set embeddings using the configured prototype method
        (mean, trimmed_mean, or weighted).

        If a projection head is set, applies it to embeddings before
        computing prototypes.

        Returns:
            Tuple containing:
                - prototypes: 2D array of shape (n_classes, embedding_dim)
                  with one prototype per class.
                - class_ids: 1D array of class IDs in sorted order.

        Raises:
            RuntimeError: If projection model inference fails.
        """
        class_ids = sorted(self.support_indices.keys())
        prototypes = []
        train_emb = self._get_projected_train_embeddings()

        for class_id in class_ids:
            indices = self.support_indices[class_id]

            if len(indices) == 0:
                out_dim = (
                    train_emb.shape[1]
                    if len(train_emb) > 0
                    else self.extractor.embedding_dim
                )
                prototypes.append(np.zeros(out_dim))
                continue

            embeddings = train_emb[indices]
            prototype = self._compute_prototype_from_embeddings(embeddings)
            prototypes.append(prototype)

        return np.array(prototypes), np.array(class_ids)

    def _compute_prototype_from_embeddings(
        self, embeddings: npt.NDArray[np.floating[Any]]
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute a class prototype from its support embeddings.

        Uses the configured prototype method to aggregate embeddings:
            - 'mean': Simple average of all embeddings.
            - 'trimmed_mean': Exclude outliers before averaging.
            - 'weighted': Weight embeddings by similarity to centroid.

        Args:
            embeddings: Support set embeddings for one class with shape
                (N, D) where N is number of support samples.

        Returns:
            1D prototype vector of shape (D,) representing the class.

        Raises:
            ValueError: If embeddings is not 2-dimensional.
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, D)")

        n = int(embeddings.shape[0])
        if n == 0:
            return np.zeros(self.extractor.embedding_dim, dtype=embeddings.dtype)
        if n == 1 or self.prototype_method == "mean":
            return embeddings.mean(axis=0)

        # Use cosine similarity to an initial centroid to detect outliers/inliers.
        centroid = embeddings.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        emb_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
        sims = emb_norm @ centroid_norm

        trim_k = min(self.prototype_trim_k, max(0, n - 1))
        if self.prototype_method == "trimmed_mean":
            if trim_k > 0 and n > trim_k:
                keep = np.argsort(sims)[trim_k:]
                return embeddings[keep].mean(axis=0)
            return embeddings.mean(axis=0)

        if self.prototype_method == "weighted":
            if trim_k > 0 and n > trim_k:
                keep = np.argsort(sims)[trim_k:]
                embeddings = embeddings[keep]
                sims = sims[keep]
                n = int(embeddings.shape[0])
            weights = np.maximum(sims, 0.0) ** float(self.prototype_weight_power)
            if float(weights.sum()) <= 1e-12:
                return embeddings.mean(axis=0)
            weights = weights / (weights.sum() + 1e-8)
            return (embeddings * weights.reshape(n, 1)).sum(axis=0)
        return embeddings.mean(axis=0)

    def _update_per_class_stats(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Update per-class accuracy statistics on validation set.

        Computes and stores accuracy, support size, pool size, and
        validation count for each class. Useful for identifying
        underperforming classes.

        Args:
            results: Optional pre-computed evaluation results dictionary
                containing 'true_labels' and 'predictions' arrays. If None,
                runs evaluation on validation set first.

        Returns:
            None. Updates self.per_class_stats in place.

        Raises:
            None: This method handles missing data gracefully.
        """
        if results is None:
            results = self.evaluate_on_val()

        for class_id in self.class_ids:
            mask = results["true_labels"] == class_id
            if mask.sum() > 0:
                class_acc = (results["predictions"][mask] == class_id).mean()
                self.per_class_stats[class_id] = {
                    "accuracy": class_acc,
                    "support_size": len(self.support_indices.get(class_id, [])),
                    "pool_size": len(self.pool_indices.get(class_id, [])),
                    "val_count": mask.sum(),
                }

    def get_support_count(self) -> int:
        """Get total number of samples in the support set.

        Counts all support samples across all classes. Useful for
        tracking support set growth during pseudo-labeling.

        Returns:
            Total count of support samples across all classes.

        Raises:
            None: This method always succeeds.
        """
        return sum(len(v) for v in self.support_indices.values())

    def get_pool_count(self) -> int:
        """Get total number of samples remaining in the unlabeled pool.

        Counts all pool samples across all classes. Useful for
        tracking pool depletion during pseudo-labeling.

        Returns:
            Total count of unlabeled pool samples across all classes.

        Raises:
            None: This method always succeeds.
        """
        return sum(len(v) for v in self.pool_indices.values())

    def evaluate_on_val(self) -> Dict[str, Any]:
        """Evaluate current prototypes on validation split from training data.

        Performs nearest-prototype classification on the validation set
        and computes standard classification metrics. Use this for monitoring
        performance during iterative pseudo-labeling.

        Does not touch the final held-out test set (ds_val).

        If a projection head is set, applies it to embeddings before evaluation.

        Returns:
            Dictionary containing:
                - accuracy: Classification accuracy (0-1).
                - precision: Macro-averaged precision.
                - recall: Macro-averaged recall.
                - f1: Macro-averaged F1 score.
                - predictions: Array of predicted class IDs.
                - true_labels: Array of ground truth labels.
                - confidences: Array of prediction confidence scores.
                - similarities: 2D array of cosine similarities to prototypes.

        Raises:
            RuntimeError: If projection model inference fails.
        """
        train_emb = self._get_projected_train_embeddings()
        return self._evaluate_on_indices(
            self.val_flat, self._val_labels, train_emb, desc="Evaluating on validation"
        )

    def _evaluate_on_indices(
        self,
        indices: npt.NDArray[np.int64],
        labels: npt.NDArray[np.int64],
        embeddings: npt.NDArray[np.floating[Any]],
        desc: str = "Evaluating",
    ) -> Dict[str, Any]:
        """Evaluate classification on a specific set of indices.

        Performs nearest-prototype classification using cosine similarity
        and computes evaluation metrics.

        Args:
            indices: Array of dataset indices to evaluate.
            labels: Array of ground truth labels for the indices. Must
                have same length as indices.
            embeddings: Full embeddings array to index into. Shape should
                be (n_total_samples, embedding_dim).
            desc: Description string for logging. Default is "Evaluating".

        Returns:
            Dictionary containing:
                - accuracy: Classification accuracy (0-1).
                - precision: Macro-averaged precision.
                - recall: Macro-averaged recall.
                - f1: Macro-averaged F1 score.
                - predictions: Array of predicted class IDs.
                - true_labels: Array of ground truth labels.
                - confidences: Array of prediction confidence scores.
                - similarities: 2D array of cosine similarities.

        Raises:
            IndexError: If any index is out of bounds for embeddings.
        """
        eval_embeddings = embeddings[indices]
        prototypes_norm = self.prototypes / (
            np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8
        )
        eval_emb_norm = eval_embeddings / (
            np.linalg.norm(eval_embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = eval_emb_norm @ prototypes_norm.T
        logits = similarities * 10
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        pred_indices = np.argmax(similarities, axis=1)
        predictions = self.class_ids[pred_indices]
        confidences = np.max(probs, axis=1)

        accuracy = (predictions == labels).mean()
        precision = precision_score(
            labels, predictions, average="macro", zero_division=0
        )
        recall = recall_score(labels, predictions, average="macro", zero_division=0)
        f1 = f1_score(labels, predictions, average="macro", zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": predictions,
            "true_labels": labels,
            "confidences": confidences,
            "similarities": similarities,
        }

    def get_high_confidence_predictions(
        self,
        threshold: float = 0.8,
        max_per_class: Optional[int] = None,
        *,
        margin_threshold: Optional[float] = None,
        mutual_nn_k: Optional[int] = None,
        sim_threshold: Optional[float] = None,
        sim_margin_threshold: Optional[float] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Get high-confidence predictions from the unlabeled pool.

        Classifies all pool samples using nearest-prototype classification
        and returns those exceeding configurable thresholds, grouped by
        predicted class. Useful for identifying pseudo-labeling candidates.

        Multiple filtering options can be combined for more selective
        candidate identification:
            - Softmax probability threshold (baseline)
            - Probability margin between top-1 and top-2
            - Cosine similarity to predicted prototype
            - Similarity margin between top-1 and top-2
            - Mutual nearest-neighbor filtering

        Args:
            threshold: Minimum softmax probability for inclusion.
                Default is 0.8.
            max_per_class: Optional limit on candidates per class. If None,
                returns all qualifying candidates.
            margin_threshold: Optional minimum probability margin between
                top-1 and top-2 predictions (p1 - p2).
            mutual_nn_k: Optional K for mutual nearest-neighbor filter.
                Sample must be in top-K most similar to its predicted class.
            sim_threshold: Optional minimum cosine similarity to predicted
                prototype. More reliable than softmax for many-class problems.
            sim_margin_threshold: Optional minimum similarity gap between
                top-1 and top-2 classes.

        Returns:
            Dictionary mapping predicted class IDs to lists of candidate
            dictionaries. Each candidate contains:
                - idx: Dataset index of the sample.
                - pred_class: Predicted class ID.
                - confidence: Softmax probability.
                - true_class: Ground truth class (for evaluation).
                - similarity: Cosine similarity to predicted prototype.
                - sim_margin: Gap between top-1 and top-2 similarity.
                - margin: Probability margin (if margin_threshold used).

        Raises:
            None: Returns empty dict if pool is empty.
        """
        pool_flat = flatten_indices(self.pool_indices)
        if len(pool_flat) == 0:
            return {}

        # Get embeddings and classify (use projected embeddings if projection is set)
        train_emb = self._get_projected_train_embeddings()
        pool_embeddings = train_emb[pool_flat]

        prototypes_norm = self.prototypes / (
            np.linalg.norm(self.prototypes, axis=1, keepdims=True) + 1e-8
        )
        pool_emb_norm = pool_embeddings / (
            np.linalg.norm(pool_embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = pool_emb_norm @ prototypes_norm.T

        # Softmax
        logits = similarities * 10
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        pred_indices = np.argmax(similarities, axis=1)
        predictions = self.class_ids[pred_indices]
        confidences = np.max(probs, axis=1)

        # Raw cosine similarity to predicted prototype
        top1_similarities = np.max(similarities, axis=1)

        # Similarity margin: gap between top-1 and top-2 in embedding space
        top2_sims = np.partition(similarities, kth=-2, axis=1)[:, -2:]
        sim_top1 = np.max(top2_sims, axis=1)
        sim_top2 = np.min(top2_sims, axis=1)
        sim_margins = sim_top1 - sim_top2

        # Top-2 margin in probability space
        if margin_threshold is not None:
            top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
            p2 = np.min(top2, axis=1)
            p1 = np.max(top2, axis=1)
            margins = p1 - p2
        else:
            margins = None

        # Mutual NN filter: sample must also be close to the prototype from the prototype's viewpoint
        if mutual_nn_k is not None and int(mutual_nn_k) > 0:
            k = int(mutual_nn_k)
            n_pool = similarities.shape[0]
            k = min(k, n_pool)
            topk_by_class: Dict[int, set[int]] = {}
            for proto_col, class_id in enumerate(self.class_ids):
                col = similarities[:, proto_col]
                if k == n_pool:
                    top_pos = np.arange(n_pool, dtype=int)
                else:
                    top_pos = np.argpartition(col, kth=n_pool - k)[n_pool - k :]
                topk_by_class[int(class_id)] = set(int(pool_flat[p]) for p in top_pos)
        else:
            topk_by_class = None

        true_labels = get_labels_for_indices(self.ds_train, pool_flat)
        high_conf_mask = confidences >= threshold
        if margins is not None:
            high_conf_mask = high_conf_mask & (margins >= float(margin_threshold))
        if sim_threshold is not None:
            high_conf_mask = high_conf_mask & (
                top1_similarities >= float(sim_threshold)
            )
        if sim_margin_threshold is not None:
            high_conf_mask = high_conf_mask & (
                sim_margins >= float(sim_margin_threshold)
            )
        results = defaultdict(list)
        for i, (is_high_conf, pred_class, conf, true_label) in enumerate(
            zip(high_conf_mask, predictions, confidences, true_labels)
        ):
            if is_high_conf:
                if topk_by_class is not None:
                    ds_idx = int(pool_flat[i])
                    if ds_idx not in topk_by_class.get(int(pred_class), set()):
                        continue
                results[int(pred_class)].append(
                    {
                        "idx": int(pool_flat[i]),
                        "pred_class": int(pred_class),
                        "confidence": float(conf),
                        "true_class": int(true_label),
                        "margin": float(margins[i]) if margins is not None else None,
                        "similarity": float(top1_similarities[i]),
                        "sim_margin": float(sim_margins[i]),
                    }
                )

        # Sort by similarity (more reliable) then confidence, and limit per class
        for class_id in results:
            results[class_id].sort(
                key=lambda x: (x["similarity"], x["confidence"]), reverse=True
            )
            if max_per_class is not None:
                results[class_id] = results[class_id][:max_per_class]

        return dict(results)

    def add_to_support(self, indices: List[int], class_id: int) -> None:
        """Add samples to the support set for a specific class.

        Adds the specified indices to the support set and removes them
        from the unlabeled pool. Automatically recomputes prototypes to
        reflect the expanded support set.

        Note: Pool indices are keyed by true class, so removal is done
        globally across all pool classes to handle pseudo-label mismatches.

        Args:
            indices: List of dataset indices to add to support.
            class_id: Class ID to add the samples to. This is the
                predicted class for pseudo-labeled samples.

        Returns:
            None. Modifies support_indices and pool_indices in place.

        Raises:
            None: Empty indices list is handled gracefully.
        """
        indices = np.array(indices, dtype=np.int64)
        if class_id not in self.support_indices:
            self.support_indices[class_id] = indices
        else:
            self.support_indices[class_id] = np.concatenate(
                [self.support_indices[class_id], indices]
            )
        if len(indices) > 0:
            remove_arr = np.asarray(indices, dtype=np.int64)
            for pool_cid, pool_arr in list(self.pool_indices.items()):
                pool_arr = np.asarray(pool_arr, dtype=np.int64)
                if pool_arr.size == 0:
                    continue
                self.pool_indices[pool_cid] = pool_arr[~np.isin(pool_arr, remove_arr)]

        self.prototypes, self.class_ids = self._compute_prototypes_from_cache()
        self.class_id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}

    def run_iteration(
        self,
        confidence_threshold: float = 0.8,
        max_per_class: int = 5,
        use_true_labels: bool = True,
        *,
        margin_threshold: Optional[float] = None,
        mutual_nn_k: Optional[int] = None,
        sim_threshold: Optional[float] = None,
        sim_margin_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run one iteration of pseudo-labeling.

        Identifies high-confidence predictions from the pool and adds them
        to the support set. Can operate in two modes:
            - Simulation (use_true_labels=True): Only accepts correct predictions.
              Useful for measuring upper-bound performance.
            - Real (use_true_labels=False): Accepts all high-confidence predictions.
              Mimics real-world usage where true labels are unknown.

        Args:
            confidence_threshold: Minimum softmax probability for pseudo-label
                acceptance. Default is 0.8.
            max_per_class: Maximum samples to add per class per iteration.
                Prevents class imbalance. Default is 5.
            use_true_labels: If True, only accept correct predictions (simulation
                mode). If False, accept all high-confidence predictions.
            margin_threshold: Optional minimum probability margin (top1 - top2).
            mutual_nn_k: Optional K for mutual nearest-neighbor filter.
            sim_threshold: Optional minimum cosine similarity to prototype.
            sim_margin_threshold: Optional minimum similarity margin.

        Returns:
            Dictionary containing iteration statistics:
                - iteration: Current iteration number.
                - n_candidates: Number of candidates found.
                - n_added: Number of samples actually added.
                - n_correct: Number of correctly predicted (if use_true_labels).
                - n_wrong: Number of incorrectly predicted.
                - support_count: New support set size.
                - pool_count: Remaining pool size.
                - val_accuracy: Validation accuracy after update.
                - val_f1: Validation F1 score after update.

        Raises:
            None: Handles edge cases (empty pool, no candidates) gracefully.
        """
        self.iteration += 1

        # Get high-confidence predictions
        high_conf = self.get_high_confidence_predictions(
            threshold=confidence_threshold,
            max_per_class=max_per_class,
            margin_threshold=margin_threshold,
            mutual_nn_k=mutual_nn_k,
            sim_threshold=sim_threshold,
            sim_margin_threshold=sim_margin_threshold,
        )

        n_candidates = sum(len(v) for v in high_conf.values())
        n_added = 0
        n_correct = 0
        n_wrong = 0
        accepted_samples = []

        # Collect additions per class so as to recompute prototypes once per class
        to_add_by_class = defaultdict(list)

        for class_id, candidates in high_conf.items():
            for cand in candidates:
                pred_ok = cand["pred_class"] == cand["true_class"]

                if use_true_labels and not pred_ok:
                    n_wrong += 1
                    continue
                to_add_by_class[int(class_id)].append(int(cand["idx"]))
                n_added += 1
                if pred_ok:
                    n_correct += 1
                else:
                    n_wrong += 1

                accepted_samples.append(
                    {
                        "idx": int(cand["idx"]),
                        "pred_class": int(cand["pred_class"]),
                        "true_class": int(cand["true_class"]),
                        "confidence": float(cand["confidence"]),
                        "similarity": float(cand.get("similarity", 0.0)),
                        "sim_margin": float(cand.get("sim_margin", 0.0)),
                    }
                )

        for class_id, idxs in to_add_by_class.items():
            if idxs:
                self.add_to_support(idxs, class_id)

        # Evaluate after adding
        val_results = self.evaluate_on_val()
        self._update_per_class_stats(val_results)

        iteration_stats = {
            "iteration": self.iteration,
            "threshold_used": float(confidence_threshold),
            "margin_threshold_used": (
                float(margin_threshold) if margin_threshold is not None else None
            ),
            "mutual_nn_k_used": int(mutual_nn_k) if mutual_nn_k is not None else None,
            "sim_threshold_used": (
                float(sim_threshold) if sim_threshold is not None else None
            ),
            "sim_margin_threshold_used": (
                float(sim_margin_threshold)
                if sim_margin_threshold is not None
                else None
            ),
            "n_candidates": int(n_candidates),
            "n_added": int(n_added),
            "samples_added": int(n_added),
            "n_correct": int(n_correct),
            "n_wrong": int(n_wrong),
            "added_samples": accepted_samples,
            "support_count": int(self.get_support_count()),
            "support_size": int(self.get_support_count()),
            "pool_count": int(self.get_pool_count()),
            "pool_size": int(self.get_pool_count()),
            "val_accuracy": float(val_results["accuracy"]),
            "val_precision": float(val_results.get("precision", 0.0)),
            "val_recall": float(val_results.get("recall", 0.0)),
            "val_f1": float(val_results.get("f1", 0.0)),
            "accuracy_after": float(val_results["accuracy"]),
            "precision_after": float(val_results.get("precision", 0.0)),
            "recall_after": float(val_results.get("recall", 0.0)),
            "f1_after": float(val_results.get("f1", 0.0)),
        }

        self.history.append(iteration_stats)
        return iteration_stats

    def run_auto_pseudo_labeling(
        self,
        target_accuracy: float = 0.70,
        initial_threshold: float = 0.8,
        min_threshold: float = 0.2,
        threshold_decay: float = 0.1,
        margin_threshold: Optional[float] = None,
        min_margin_threshold: float = 0.0,
        margin_decay: float = 0.0,
        sim_threshold: Optional[float] = None,
        min_sim_threshold: float = -1.0,
        sim_decay: float = 0.0,
        sim_margin_threshold: Optional[float] = None,
        min_sim_margin_threshold: float = 0.0,
        sim_margin_decay: float = 0.0,
        mutual_nn_k: Optional[int] = None,
        max_iterations: int = 50,
        max_per_class: int = 5,
        use_true_labels: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Automatically run pseudo-labeling until target accuracy is reached.

        Iteratively runs pseudo-labeling iterations, adding high-confidence
        predictions to the support set. Automatically lowers thresholds when
        no candidates are found. Stops when target accuracy is reached, pool
        is exhausted, or max iterations exceeded.

        Args:
            target_accuracy: Target validation accuracy to stop at (0-1).
                Default is 0.70 (70%).
            initial_threshold: Starting softmax probability threshold.
                Default is 0.8.
            min_threshold: Minimum threshold before considering other options.
                Default is 0.2.
            threshold_decay: Amount to reduce threshold when stuck.
                Default is 0.1.
            margin_threshold: Optional minimum probability margin for acceptance.
            min_margin_threshold: Minimum margin threshold. Default is 0.0.
            margin_decay: Amount to reduce margin when stuck. 0 disables.
            sim_threshold: Optional minimum cosine similarity for acceptance.
            min_sim_threshold: Minimum sim threshold. Default is -1.0.
            sim_decay: Amount to reduce sim threshold when stuck. 0 disables.
            sim_margin_threshold: Optional minimum similarity gap for acceptance.
            min_sim_margin_threshold: Minimum sim margin. Default is 0.0.
            sim_margin_decay: Amount to reduce sim margin when stuck. 0 disables.
            mutual_nn_k: Optional K for mutual nearest-neighbor filter.
            max_iterations: Maximum iterations to run. Default is 50.
            max_per_class: Maximum samples to add per class per iteration.
                Default is 5.
            use_true_labels: If True, simulation mode (only correct predictions).
            verbose: Whether to print progress updates. Default is True.

        Returns:
            Dictionary containing:
                - final_accuracy: Final validation accuracy.
                - final_precision: Final macro precision.
                - final_recall: Final macro recall.
                - final_f1: Final macro F1 score.
                - iterations: Total iterations run.
                - total_added: Total samples added to support.
                - target_reached: Whether target accuracy was achieved.

        Raises:
            None: Handles all edge cases gracefully.
        """
        print(f"\n{'='*70}")
        print("AUTOMATIC PSEUDO-LABELING")
        print(f"{'='*70}")
        print(f"Target accuracy: {target_accuracy*100:.0f}%")
        print(f"Initial threshold: {initial_threshold}")
        print(f"Min threshold: {min_threshold}")
        if margin_threshold is not None:
            print(
                f"Margin threshold: {float(margin_threshold):.3f} (min {float(min_margin_threshold):.3f}, decay {float(margin_decay):.3f})"
            )
        if sim_threshold is not None:
            print(
                f"Sim threshold: {float(sim_threshold):.3f} (min {float(min_sim_threshold):.3f}, decay {float(sim_decay):.3f})"
            )
        if sim_margin_threshold is not None:
            print(
                f"Sim margin: {float(sim_margin_threshold):.3f} (min {float(min_sim_margin_threshold):.3f}, decay {float(sim_margin_decay):.3f})"
            )
        if mutual_nn_k is not None:
            print(f"Mutual-NN K: {int(mutual_nn_k)}")
        print(f"Max iterations: {max_iterations}")
        print(f"Max per class: {max_per_class}")
        print(
            f"Mode: {'Simulation (uses true labels)' if use_true_labels else 'Real (no true labels)'}"
        )
        print(f"{'='*70}\n")

        current_threshold = float(initial_threshold)
        current_margin = (
            float(margin_threshold) if margin_threshold is not None else None
        )
        current_sim = float(sim_threshold) if sim_threshold is not None else None
        current_sim_margin = (
            float(sim_margin_threshold) if sim_margin_threshold is not None else None
        )
        iterations_without_candidates = 0
        initial_results = self.evaluate_on_val()
        current_accuracy = float(initial_results["accuracy"])
        current_precision = float(initial_results.get("precision", 0.0))
        current_recall = float(initial_results.get("recall", 0.0))
        current_f1 = float(initial_results.get("f1", 0.0))

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
                print(f"\nTARGET REACHED. Accuracy: {current_accuracy*100:.2f}%")
                break

            if self.get_pool_count() == 0:
                print(f"\nPool exhausted. No more samples to add.")
                break

            stats = self.run_iteration(
                confidence_threshold=current_threshold,
                max_per_class=max_per_class,
                use_true_labels=use_true_labels,
                margin_threshold=current_margin,
                mutual_nn_k=mutual_nn_k,
                sim_threshold=current_sim,
                sim_margin_threshold=current_sim_margin,
            )

            current_accuracy = float(
                stats.get("accuracy_after", stats.get("val_accuracy", 0.0))
            )
            current_precision = float(
                stats.get("precision_after", stats.get("val_precision", 0.0))
            )
            current_recall = float(
                stats.get("recall_after", stats.get("val_recall", 0.0))
            )
            current_f1 = float(stats.get("f1_after", stats.get("val_f1", 0.0)))

            if verbose:
                msg = (
                    f"Iter {stats['iteration']:3d} | "
                    f"Thresh: {current_threshold:.2f} | "
                )
                if current_margin is not None:
                    msg += f"Margin: {current_margin:.3f} | "
                if current_sim is not None:
                    msg += f"Sim: {current_sim:.3f} | "
                if current_sim_margin is not None:
                    msg += f"SimΔ: {current_sim_margin:.3f} | "
                msg += (
                    f"Added: {stats['n_added']:3d} | "
                    f"Acc: {current_accuracy*100:.2f}% | "
                    f"F1: {current_f1*100:.2f}% | "
                    f"Support: {stats['support_count']} | "
                    f"Pool: {stats['pool_count']}"
                )
                print(msg)

            no_progress = (stats["n_candidates"] == 0) or (
                use_true_labels and stats["n_added"] == 0
            )

            if no_progress:
                iterations_without_candidates += 1
                if current_threshold > min_threshold:
                    current_threshold = max(
                        min_threshold, current_threshold - threshold_decay
                    )
                    print(f"   → Lowering threshold to {current_threshold:.2f}")
                elif (
                    current_margin is not None
                    and float(margin_decay) > 0
                    and current_margin > float(min_margin_threshold)
                ):
                    current_margin = max(
                        float(min_margin_threshold),
                        current_margin - float(margin_decay),
                    )
                    print(f"   → Lowering margin to {current_margin:.3f}")
                elif (
                    current_sim is not None
                    and float(sim_decay) > 0
                    and current_sim > float(min_sim_threshold)
                ):
                    current_sim = max(
                        float(min_sim_threshold), current_sim - float(sim_decay)
                    )
                    print(f"   → Lowering sim threshold to {current_sim:.3f}")
                elif (
                    current_sim_margin is not None
                    and float(sim_margin_decay) > 0
                    and current_sim_margin > float(min_sim_margin_threshold)
                ):
                    current_sim_margin = max(
                        float(min_sim_margin_threshold),
                        current_sim_margin - float(sim_margin_decay),
                    )
                    print(f"   → Lowering sim margin to {current_sim_margin:.3f}")
                else:
                    print(f"   → At minimum threshold, no more candidates")
                    if iterations_without_candidates >= 3:
                        print(
                            "   → Stopping (3 consecutive iterations without candidates)"
                        )
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

        total_added = sum(
            h.get("samples_added", h.get("n_added", 0)) for h in self.history
        )
        total_correct = sum(h.get("n_correct", 0) for h in self.history)
        total_wrong = sum(h.get("n_wrong", 0) for h in self.history)

        print(f"\nTotal samples added: {total_added}")
        if use_true_labels:
            print(f"  Correct: {total_correct}")
            print(f"  Wrong (rejected): {total_wrong}")

        if current_accuracy >= target_accuracy:
            print(f"\nSUCCESS: Target accuracy of {target_accuracy*100:.0f}% reached!")
        else:
            print(f"\nTarget not reached. Current: {current_accuracy*100:.2f}%")

        print(f"{'='*70}")

        return {
            "final_accuracy": current_accuracy,
            "final_precision": current_precision,
            "final_recall": current_recall,
            "final_f1": current_f1,
            "iterations": self.iteration,
            "total_added": total_added,
            "target_reached": current_accuracy >= target_accuracy,
        }

    def save_state(self) -> Dict[str, Any]:
        """Save current experiment state for potential rollback.

        Creates a deep copy of all mutable experiment state, allowing
        restoration to this point if needed.

        Returns:
            Dictionary containing copies of:
                - support_indices: Current support set indices per class.
                - pool_indices: Current pool indices per class.
                - prototypes: Current class prototype embeddings.
                - class_ids: Array of class IDs.
                - history: List of iteration statistics.
                - iteration: Current iteration count.
                - prototype_method: Current prototype computation method.
                - prototype_trim_k: Trimming parameter.
                - prototype_weight_power: Weighting parameter.

        Raises:
            None: This method always succeeds.
        """
        return {
            "support_indices": {k: v.copy() for k, v in self.support_indices.items()},
            "pool_indices": {k: v.copy() for k, v in self.pool_indices.items()},
            "prototypes": self.prototypes.copy(),
            "class_ids": self.class_ids.copy(),
            "history": list(self.history),
            "iteration": self.iteration,
            "prototype_method": self.prototype_method,
            "prototype_trim_k": self.prototype_trim_k,
            "prototype_weight_power": self.prototype_weight_power,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore experiment state from a saved snapshot.

        Reverts all mutable experiment state to a previously saved
        checkpoint. Useful for rolling back failed pseudo-labeling runs.

        Args:
            state: State dictionary from save_state(). Must contain all
                required keys.

        Returns:
            None. Modifies experiment state in place.

        Raises:
            KeyError: If state dictionary is missing required keys.
        """
        self.support_indices = {
            k: v.copy() for k, v in state["support_indices"].items()
        }
        self.pool_indices = {k: v.copy() for k, v in state["pool_indices"].items()}
        self.prototypes = state["prototypes"].copy()
        self.class_ids = state["class_ids"].copy()
        self.class_id_to_idx = {cid: i for i, cid in enumerate(self.class_ids)}
        self.history = list(state["history"])
        self.iteration = state["iteration"]
        self.prototype_method = state["prototype_method"]
        self.prototype_trim_k = state["prototype_trim_k"]
        self.prototype_weight_power = state["prototype_weight_power"]


def spot_check_pseudo_labels(
    experiment: FewShotExperiment, n_samples: int = 10, seed: Optional[int] = None
) -> Dict[str, Any]:
    """Display a random sample of pseudo-labels for visual verification.

    Randomly selects samples from all pseudo-labeled additions and displays
    them alongside their class reference images in a matplotlib grid.
    Compares against ground truth to estimate pseudo-labeling noise rate.

    Args:
        experiment: FewShotExperiment instance with pseudo-labeling history.
            Must have run at least one pseudo-labeling iteration.
        n_samples: Number of random samples to display for verification.
            Default is 10.
        seed: Optional random seed for reproducible sampling. If None,
            uses current random state.

    Returns:
        Dictionary containing:
            - n_checked: Number of samples checked.
            - n_correct: Number of correctly pseudo-labeled samples.
            - error_rate: Fraction of incorrectly labeled samples (0-1).
            - samples: List of sample dictionaries with full details.
        Returns empty dict if no pseudo-labeling has occurred.

    Raises:
        None: Returns empty dict and prints message if no history exists.
    """
    if not experiment.history:
        print("No pseudo-labeling iterations have been run yet.")
        return {}

    all_added = []
    for h in experiment.history:
        added_samples = h.get("added_samples", [])
        all_added.extend(added_samples)

    if not all_added:
        print("No samples were added during pseudo-labeling.")
        return {}

    if seed is not None:
        np.random.seed(seed)

    n_to_check = min(n_samples, len(all_added))
    sample_indices = np.random.choice(len(all_added), size=n_to_check, replace=False)
    samples_to_check = [all_added[i] for i in sample_indices]

    print(f"{'='*70}")
    print(
        f"Spot check: {n_to_check} Random Pseudo-Labels (out of {len(all_added)} total)"
    )
    print(f"{'='*70}")

    n_cols = min(5, n_to_check)
    n_rows = (n_to_check + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols * 2, figsize=(4 * n_cols, 4 * n_rows), squeeze=False
    )

    correct_count = 0

    for i, sample in enumerate(samples_to_check):
        row = i // n_cols
        col = (i % n_cols) * 2

        pred_class = sample["pred_class"]
        true_class = sample.get("true_class", pred_class)
        is_correct = pred_class == true_class
        if is_correct:
            correct_count += 1

        ax_ref = axes[row, col]
        support_indices = experiment.support_indices.get(pred_class, [])
        if len(support_indices) > 0:
            ref_idx = int(support_indices[0])
            ref_img = experiment.ds_train["images"][ref_idx].numpy()
            ref_img = experiment.extractor._apply_preprocessing(
                ref_img, experiment.ds_train, ref_idx
            )
            ax_ref.imshow(ref_img)
        ax_ref.set_title(
            f"REF: Class {pred_class}", fontsize=9, color="green", fontweight="bold"
        )
        ax_ref.axis("off")
        for spine in ax_ref.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("green")
            spine.set_linewidth(3)
        ax_cand = axes[row, col + 1]
        cand_idx = int(sample["idx"])
        cand_img = experiment.ds_train["images"][cand_idx].numpy()
        cand_img = experiment.extractor._apply_preprocessing(
            cand_img, experiment.ds_train, cand_idx
        )
        ax_cand.imshow(cand_img)
        border_color = "green" if is_correct else "red"
        symbol = "✓" if is_correct else "x"
        ax_cand.set_title(
            f"{symbol} Conf: {sample['confidence']:.0%}\nTrue: {true_class}",
            fontsize=9,
            color=border_color,
            fontweight="bold",
        )
        ax_cand.axis("off")
        for spine in ax_cand.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    for i in range(n_to_check, n_rows * n_cols):
        row = i // n_cols
        col = (i % n_cols) * 2
        axes[row, col].set_visible(False)
        axes[row, col + 1].set_visible(False)

    plt.suptitle(
        "Spot-Check: Reference (green) vs Added Sample (green=correct, red=wrong)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

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
        "n_checked": n_to_check,
        "n_correct": correct_count,
        "error_rate": error_rate,
        "samples": samples_to_check,
    }
