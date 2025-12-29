from __future__ import annotations
import os, random, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import hashlib
import imagehash
from PIL import Image
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple
import math
from matplotlib.patches import Rectangle
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from scipy import ndimage, stats
import logging

DeepLakeDataset = Any


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================
# IMAGE DISPLAY & ASPECT RATIO
# ============================================================


def get_aspect_percent(aspects: np.ndarray) -> pd.Series:
    """
    Categorize aspect ratios into bins and return percentage distribution.

    Args:
        aspects: Array of aspect ratios (width / height).

    Returns:
        pd.Series: Percentage of images in each category
            (Portrait, ~Square, Landscape, Very wide).
    """
    cats = pd.cut(
        aspects,
        bins=[0, 0.8, 1.2, 2.0, 100],
        labels=["Portrait", "~Square", "Landscape", "Very wide"],
    )
    return cats.value_counts().sort_index() / len(cats) * 100


def show_img(ax: plt.Axes, ds: DeepLakeDataset, idx: int, title: str) -> None:
    """
    Display a single image from a DeepLake dataset on a matplotlib axis.

    Args:
        ax: Matplotlib axis to display the image on.
        ds: DeepLake dataset with an "images" tensor.
        idx: Index of the image in the dataset.
        title: Title to display above the image.

    Returns:
        None
    """
    img = ds["images"][idx].numpy()
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ============================================================
# IMAGE QUALITY CHECKING
# ============================================================


def check_image_quality(ds: DeepLakeDataset, name: str = "dataset") -> Dict[str, Any]:
    """
    Scan an entire DeepLake dataset for corrupted, nearly-black, nearly-white,
    and low-variance images.

    Args:
        ds: DeepLake dataset with an "images" tensor field.
        name: Name used in progress / summary prints.

    Returns:
        dict with:
            - total: number of images checked
            - corrupt: count of images that could not be loaded
            - all_black: count of images with very low max intensity
            - all_white: count of images with very high min intensity
            - low_var: count of images with very low standard deviation
            - bad_examples: list of (idx, reason_string) for problematic images
            - problem_indices: sorted list of unique indices for non-corrupt problematic images
    """
    n_total = len(ds)

    corrupt = 0
    all_black = 0
    all_white = 0
    low_var = 0

    bad_examples: list[tuple[int, str]] = []
    problem_indices: set[int] = set()
    for idx, sample in tqdm(
        enumerate(ds), total=n_total, desc=f"Checking {name}", unit="img"
    ):
        try:
            img = sample["images"].numpy()
        except Exception as e:
            corrupt += 1
            bad_examples.append((idx, f"corrupt: {e}"))
            continue
        arr = img.astype(np.float32)
        max_val = float(arr.max())
        if max_val <= 1.0 + 1e-6:
            arr *= 255.0

        mn = float(arr.min())
        mx = float(arr.max())
        std = float(arr.std())

        reasons = []
        if mx <= 5:
            all_black += 1
            reasons.append(f"nearly black (min={mn:.1f}, max={mx:.1f})")
        elif mn >= 250:
            all_white += 1
            reasons.append(f"nearly white (min={mn:.1f}, max={mx:.1f})")
        elif std < 1.0:
            low_var += 1
            reasons.append(f"low variance (std={std:.3f})")

        if reasons:
            problem_indices.add(idx)
            bad_examples.append((idx, "; ".join(reasons)))
    print(f"\n{name.upper()} - Checked {n_total} images:")
    print(f"  corrupt:   {corrupt}")
    print(f"  all_black: {all_black}")
    print(f"  all_white: {all_white}")
    print(f"  low_var:   {low_var}")

    if bad_examples:
        print(f"  Examples (up to 5): {bad_examples[:5]}")
    else:
        print("No problematic images found")

    return {
        "total": n_total,
        "corrupt": corrupt,
        "all_black": all_black,
        "all_white": all_white,
        "low_var": low_var,
        "bad_examples": bad_examples,
        "problem_indices": sorted(problem_indices),
    }


def n_problem_images(q: dict) -> int:
    """
    Count total number of problematic images from quality check results.

    Args:
        q: Dictionary returned by check_image_quality() containing
            'corrupt' count and 'problem_indices' list.

    Returns:
        int: Total count of corrupt + other problematic images.
    """
    return q["corrupt"] + len(q["problem_indices"])


# ============================================================
# DUPLICATE DETECTION
# ============================================================


def compute_phash_groups(
    ds: DeepLakeDataset,
) -> Dict[str, List[int]]:
    """
    Compute perceptual-hash (pHash) duplicate groups over the ENTIRE dataset.

    Args:
        ds: DeepLake dataset with an 'images' tensor.

    Returns:
        Dict[pHash -> List[dataset_indices]] for groups with >= 2 members.
    """
    hashes: Dict[str, List[int]] = {}
    for idx, sample in tqdm(
        enumerate(ds),
        total=len(ds),
        desc="Computing perceptual hashes",
        leave=False,
    ):
        img = sample["images"].numpy()
        pil_img = Image.fromarray(img)

        phash = str(imagehash.phash(pil_img))
        hashes.setdefault(phash, []).append(idx)
    dup_groups = {k: v for k, v in hashes.items() if len(v) > 1}
    return dup_groups


def refine_with_md5(
    ds: DeepLakeDataset,
    phash_groups: Dict[str, List[int]],
) -> Tuple[List[Tuple[int, int]], Dict[str, List[int]]]:
    """
    Refine pHash-based duplicate groups using MD5 hashes.

    For each pHash group, images that share the same MD5 are exact duplicates,
    while images with the same pHash but different MD5 are near-duplicates.

    Args:
        ds: DeepLake dataset with 'images' tensor.
        phash_groups: Dictionary mapping pHash string to list of dataset indices
            (groups with len >= 2).

    Returns:
        Tuple containing:
            - exact_pairs: List of (idx_a, idx_b) tuples for exact duplicates.
            - near_groups: Dictionary mapping pHash to list of indices for
                visually similar but not MD5-identical images (len >= 2).
    """
    exact_pairs: List[Tuple[int, int]] = []
    near_groups: Dict[str, List[int]] = {}

    for phash, idxs in tqdm(
        phash_groups.items(),
        desc="Refining duplicate groups with MD5",
        leave=False,
    ):
        md5_map: Dict[str, List[int]] = {}

        for idx in idxs:
            img = ds["images"][idx].numpy()
            md5 = hashlib.md5(img.tobytes()).hexdigest()
            md5_map.setdefault(md5, []).append(idx)

        exact_indices: set[int] = set()
        for md5, md5_idxs in md5_map.items():
            if len(md5_idxs) > 1:
                for i in range(len(md5_idxs)):
                    for j in range(i + 1, len(md5_idxs)):
                        exact_pairs.append((md5_idxs[i], md5_idxs[j]))
                exact_indices.update(md5_idxs)

        near = [idx for idx in idxs if idx not in exact_indices]
        if len(near) > 1:
            near_groups[phash] = near

    return exact_pairs, near_groups


def show_duplicate_groups(
    ds: DeepLakeDataset,
    near_groups: Dict[str, List[int]],
    group_type: str = "all",
    title_prefix: str = "",
    id_to_name: Dict[int, str] | None = None,
    n_groups: int = 5,
) -> None:
    """
    Visualize near-duplicate image groups with their labels.

    Shows dataset index and label at the top of each subplot; uses
    constrained_layout to avoid clipping.

    Args:
        ds: DeepLake dataset with 'images' and 'labels' tensors.
        near_groups: Dictionary mapping pHash to list of dataset indices.
        group_type: Filter type - 'all', 'consistent' (same label),
            or 'inconsistent' (different labels).
        title_prefix: Prefix string for figure titles.
        id_to_name: Optional mapping from class_id to human-readable name.
        n_groups: Maximum number of groups to display.

    Returns:
        None
    """

    consistent_groups = []
    inconsistent_groups = []

    for phash, indices in near_groups.items():
        labels = []
        for idx in indices:
            try:
                label_arr = ds["labels"][int(idx)].numpy()
                label_id = int(label_arr.flat[0])
                labels.append(label_id)
            except Exception:
                labels.append(-1)

        if len(set(labels)) == 1:
            consistent_groups.append((phash, indices, labels))
        else:
            inconsistent_groups.append((phash, indices, labels))

    if group_type == "consistent":
        groups_to_show = consistent_groups
        header = "CONSISTENT LABELS (same pHash, same label)"
    elif group_type == "inconsistent":
        groups_to_show = inconsistent_groups
        header = "INCONSISTENT LABELS (same pHash, different labels)"
    else:
        groups_to_show = consistent_groups + inconsistent_groups
        header = "ALL GROUPS"

    total_available = len(groups_to_show)
    print(
        f"\n{title_prefix}: {len(consistent_groups)} consistent, {len(inconsistent_groups)} inconsistent groups"
    )
    print(f"Showing: {header}")
    print(f"Displaying {min(n_groups, total_available)} of {total_available} groups\n")

    if not groups_to_show:
        print(f"No {group_type} groups to show.")
        return

    for group_idx, (phash, indices, labels) in enumerate(groups_to_show[:n_groups]):
        n_imgs = len(indices)
        cols = min(n_imgs, 3)
        rows = math.ceil(n_imgs / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(3.5 * cols, 4 * rows),
            constrained_layout=True,
        )

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()

        is_consistent = len(set(labels)) == 1
        title_color = "green" if is_consistent else "red"
        status = "Same labels" if is_consistent else "Different labels"

        fig.suptitle(
            f"{title_prefix} Group {group_idx + 1}/{min(n_groups, total_available)} | "
            f"pHash: {phash[:10]}... | {status}",
            fontsize=11,
            fontweight="bold",
            color=title_color,
        )

        for i, (idx, label_id) in enumerate(zip(indices, labels)):
            ax = axes[i]
            try:
                img = ds["images"][int(idx)].numpy()
                label_txt = (
                    id_to_name.get(label_id, f"ID: {label_id}")
                    if id_to_name
                    else f"ID: {label_id}"
                )
                if len(label_txt) > 40:
                    label_txt = label_txt[:37] + "..."
                ax.imshow(img)
                ax.set_title(f"idx={idx}\n{label_txt}", fontsize=8, pad=4)
            except Exception:
                ax.text(
                    0.5,
                    0.5,
                    f"Load error\nidx={idx}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            ax.axis("off")

        for i in range(n_imgs, len(axes)):
            axes[i].axis("off")

        plt.show()
        print()


def get_same_label_duplicates_to_drop(
    ds: DeepLakeDataset, near_groups: Dict[str, List[int]]
) -> set[int]:
    """
    Find indices to drop from duplicate groups where all images share the same label.

    Keeps the first image in each group and marks the rest for removal.

    Args:
        ds: DeepLake dataset with a 'labels' tensor.
        near_groups: Dictionary mapping pHash to list of dataset indices
            representing near-duplicate groups.

    Returns:
        set[int]: Set of dataset indices that should be dropped.
    """
    to_drop = set()

    for phash, indices in near_groups.items():
        labels = []
        for idx in indices:
            try:
                label_arr = ds["labels"][int(idx)].numpy()
                labels.append(int(label_arr.flat[0]))
            except:
                labels.append(-1)

        if len(set(labels)) == 1:
            to_drop.update(indices[1:])

    return to_drop


# ============================================================
# LABEL INDEXING
# ============================================================


def build_label_index(ds: DeepLakeDataset) -> Dict[int, np.ndarray]:
    """
    Build a mapping from class label to array of dataset indices.

    Useful for avoiding repeated full-dataset scans when sampling rare/common
    classes or creating few-shot subsets.

    Args:
        ds: DeepLake dataset with a 'labels' tensor.

    Returns:
        Dict[int, np.ndarray]: Mapping from class_id to array of dataset indices
            belonging to that class.
    """
    label_to_idxs: Dict[int, list[int]] = defaultdict(list)
    for i, sample in enumerate(ds):
        label = int(sample["labels"].numpy()[0])
        label_to_idxs[label].append(int(i))
    return {k: np.array(v, dtype=np.int64) for k, v in label_to_idxs.items()}


def save_label_index(
    label_index: Dict[int, np.ndarray], path: str | os.PathLike
) -> None:
    """
    Persist the label index to disk as a compressed .npz file.

    Args:
        label_index: Dictionary mapping class_id to array of dataset indices.
        path: File path to save the label index (should end in .npz).

    Returns:
        None
    """
    np.savez_compressed(path, **{str(k): v for k, v in label_index.items()})


def load_label_index(path: str | os.PathLike) -> Dict[int, np.ndarray]:
    """
    Load a label index previously saved by save_label_index.

    Args:
        path: File path to the saved .npz label index file.

    Returns:
        Dict[int, np.ndarray]: Mapping from class_id to array of dataset indices.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    data = np.load(path)
    return {int(k): data[k] for k in data.files}


# ============================================================
# DIMENSION ANALYSIS
# ============================================================


def analyze_dimensions(
    ds: DeepLakeDataset, sample_size: int = 300
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample images from a DeepLake dataset and compute dimension statistics.

    Args:
        ds: DeepLake dataset with an 'images' tensor.
        sample_size: Maximum number of images to sample for analysis.

    Returns:
        Tuple containing:
            - heights: Array of image heights in pixels.
            - widths: Array of image widths in pixels.
            - aspects: Array of aspect ratios (width / height).
            - areas: Array of image areas (height * width).
    """
    H, W, aspects, areas = [], [], [], []

    sample_n = min(len(ds), sample_size)
    for i, sample in tqdm(enumerate(ds), total=sample_n, desc="Analyzing dimensions"):
        if i >= sample_n:
            break
        img = sample["images"].numpy()
        h, w = img.shape[:2]
        H.append(h)
        W.append(w)
        aspects.append(w / h)
        areas.append(h * w)

    return np.array(H), np.array(W), np.array(aspects), np.array(areas)


# ============================================================
# CLASS BALANCE METRICS
# ============================================================


def counts_from_index(idx_map: Dict[int, np.ndarray]) -> Dict[int, int]:
    """
    Convert a label index to class counts.

    Args:
        idx_map: Dictionary mapping class_id to array of dataset indices.

    Returns:
        Dict[int, int]: Mapping from class_id to count of samples in that class.
    """
    return {cls: len(idxs) for cls, idxs in idx_map.items()}


def gini(x: np.ndarray | list) -> float:
    """
    Compute the Gini coefficient for measuring class imbalance.

    A Gini coefficient of 0 indicates perfect equality (all classes have
    the same count), while 1 indicates maximum inequality.

    Args:
        x: Array-like of class counts or values.

    Returns:
        float: Gini coefficient in range [0, 1].
    """
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-9
    x = np.sort(x)
    n = len(x)
    return 2 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x)) - (n + 1) / n


# ============================================================
# VISUALIZATION HELPERS
# ============================================================


def show_samples_by_class(
    ds: DeepLakeDataset,
    class_ids: List[int],
    title: str,
    label_index: Dict[int, np.ndarray] | None = None,
    per_class: int = 2,
    seed: int = 42,
    show_boxes: bool = True,
    id_to_name: Dict[int, str] | None = None,
) -> None:
    """
    Display sample images from specified classes in a grid.

    Args:
        ds: DeepLake dataset with 'images', 'labels', and optionally 'boxes' tensors.
        class_ids: List of class IDs to display samples from.
        title: Title for the figure.
        label_index: Optional pre-computed label index for faster sampling.
            If None, iterates through the dataset to find samples.
        per_class: Number of samples to show per class.
        seed: Random seed for reproducible sampling.
        show_boxes: Whether to overlay bounding boxes on images.
        id_to_name: Optional mapping from class_id to human-readable name.

    Returns:
        None
    """
    import math

    rng = np.random.default_rng(seed)
    n = len(class_ids) * per_class
    cols = per_class
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True
    )
    axes = np.array(axes).reshape(rows, cols)
    ax_flat = axes.flatten()

    pos = 0
    for ci, cid in enumerate(class_ids):
        if label_index and cid in label_index:
            choices = label_index[cid]
        else:
            choices = [
                i
                for i, sample in enumerate(ds)
                if int(sample["labels"].numpy().flat[0]) == cid
            ]
        if len(choices) == 0:
            continue
        pick = rng.choice(choices, size=min(per_class, len(choices)), replace=False)
        for idx in pick:
            ax = ax_flat[pos]
            img = ds["images"][int(idx)].numpy()
            ax.imshow(img)
            if show_boxes and "boxes" in ds.tensors:
                bbox = resolve_bbox(ds, int(idx))
                if bbox:
                    x1, y1, x2, y2 = bbox
                    ax.add_patch(
                        Rectangle(
                            (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", lw=2
                        )
                    )
            name = id_to_name.get(cid, cid) if id_to_name else cid
            ax.set_title(f"{name} (id {cid})", fontsize=10)
            ax.axis("off")
            pos += 1

    for k in range(pos, len(ax_flat)):
        ax_flat[k].axis("off")

    fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=14)
    plt.show()


def compare_resize_strategies(
    ds: DeepLakeDataset,
    indices: List[int],
    id_to_name: Dict[int, str] | None = None,
    img_size: int = 224,
    figsize_scale: float = 2.5,
) -> None:
    """
    Compare three resize strategies side-by-side on multiple images.

    Visualizes the following columns:
        0. Original image
        1. Resize (squish to square)
        2. LongestMaxSize + Pad (aspect-preserving with reflect padding)
        3. Bbox Crop + Pad (crop to bbox, then pad)

    Args:
        ds: DeepLake dataset with 'images', 'labels', and optionally 'boxes' tensors.
        indices: List of dataset indices to visualize.
        id_to_name: Optional mapping from class_id to human-readable name.
        img_size: Target size for the square outputs.
        figsize_scale: Scale factor for figure size.

    Returns:
        None
    """
    transform_squish = A.Compose(
        [
            A.Resize(img_size, img_size),
        ]
    )

    transform_pad = A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
        ]
    )

    n_images = len(indices)
    if n_images == 0:
        print("No indices provided to compare_resize_strategies.")
        return

    fig, axes = plt.subplots(
        n_images,
        4,
        figsize=(figsize_scale * 4, figsize_scale * n_images),
    )

    if n_images == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Resize Strategy Comparison: Squish vs Pad vs Bbox Crop",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    column_titles = ["Original", "Resize (Squish)", "Pad (Reflect)", "Bbox Crop + Pad"]

    for row, idx in enumerate(indices):
        idx = int(idx)
        img = ds["images"][idx].numpy()
        h, w = img.shape[:2]
        aspect = w / h if h > 0 else float("nan")

        try:
            label_arr = ds["labels"][idx].numpy()
            label_id = int(label_arr.flat[0])
        except Exception:
            label_id = -1

        if id_to_name:
            class_name = id_to_name.get(label_id, f"Class {label_id}")
        else:
            class_name = f"Class {label_id}"
        class_name = str(class_name)[:25]

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(
            f"{class_name}\n{w}x{h} (AR={aspect:.2f})",
            fontsize=8,
        )
        if row == 0:
            axes[row, 0].set_title(column_titles[0], fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        squished = transform_squish(image=img)["image"]
        axes[row, 1].imshow(squished)
        if row == 0:
            axes[row, 1].set_title(column_titles[1], fontsize=10, fontweight="bold")
        axes[row, 1].axis("off")

        padded = transform_pad(image=img)["image"]
        axes[row, 2].imshow(padded)
        if row == 0:
            axes[row, 2].set_title(column_titles[2], fontsize=10, fontweight="bold")
        axes[row, 2].axis("off")

        cropped = apply_bbox_crop(img, ds, idx, padding_ratio=0.15)
        cropped_padded = transform_pad(image=cropped)["image"]
        axes[row, 3].imshow(cropped_padded)
        if row == 0:
            axes[row, 3].set_title(column_titles[3], fontsize=10, fontweight="bold")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# BOUNDING BOX ANALYSIS
# ============================================================
def resolve_bbox(
    ds: DeepLakeDataset, idx: int
) -> Tuple[float, float, float, float] | None:
    """
    Extract and normalize bounding box to pixel-space (x1, y1, x2, y2).

    Handles multiple bbox formats: (x, y, w, h), (x1, y1, x2, y2) in pixels,
    and normalized corner coordinates.

    Args:
        ds: DeepLake dataset with 'images' and 'boxes' tensors.
        idx: Index of the sample in the dataset.

    Returns:
        Tuple[float, float, float, float] | None: Bounding box as
            (x1, y1, x2, y2) clipped to image bounds, or None if invalid.
    """
    img = ds["images"][idx].numpy()
    h, w = img.shape[:2]
    box = ds["boxes"][idx].numpy().astype(float).squeeze()
    if box.shape[-1] != 4:
        return None
    x1, y1, x2, y2 = box

    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
    else:
        width, height = x2, y2
        is_wh = (
            width > 0
            and height > 0
            and x1 + width <= w + 1e-3
            and y1 + height <= h + 1e-3
        )
        if is_wh:
            x2 = x1 + width
            y2 = y1 + height

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


# Alias for backward compatibility - identical to resolve_bbox
resolve_bbox_xywh_or_xyxy = resolve_bbox


def bbox_coverage_simple(
    ds: DeepLakeDataset, n_samples: int = 2000, seed: int = 42
) -> np.ndarray:
    """
    Sample random images and compute bounding box coverage ratios.

    Coverage is defined as the fraction of image area covered by the
    bounding box.

    Args:
        ds: DeepLake dataset with 'images' and 'boxes' tensors.
        n_samples: Number of random samples to analyze.
        seed: Random seed for reproducible sampling.

    Returns:
        np.ndarray: Array of coverage ratios (0 to 1) for each valid sample.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    coverages = []
    for idx in indices:
        bbox = resolve_bbox_xywh_or_xyxy(ds, int(idx))
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        img = ds["images"][int(idx)].numpy()
        h, w = img.shape[:2]
        coverages.append((x2 - x1) * (y2 - y1) / (h * w))

    return np.array(coverages)


# ============================================================
# IMAGE QUALITY METRICS
# ============================================================


def compute_image_metrics_raw(
    ds: DeepLakeDataset,
    label_index: Dict[int, np.ndarray],
    sample_per_class: int = 10,
    seed: int = 42,
    edge_percentile: int = 75,
) -> pd.DataFrame:
    """
    Compute quality and appearance metrics for a sample of images per class.

    Args:
        ds: DeepLake dataset with an 'images' tensor.
        label_index: Dictionary mapping class_id to array of dataset indices.
        sample_per_class: Number of images to sample per class.
        seed: Random seed for reproducible sampling.
        edge_percentile: Percentile threshold for Sobel edge magnitude used
            to compute edge_density. Pixels above this percentile are
            considered "strong edges". Higher values are more selective.

    Returns:
        pd.DataFrame: DataFrame with one row per image containing:
            - idx: Dataset index
            - class_id: Class label
            - h, w: Image dimensions in pixels
            - aspect_ratio: Width / height
            - brightness: Grayscale mean (0-1)
            - contrast: Grayscale standard deviation (0-1)
            - blur: Variance of Laplacian (higher = sharper)
            - r_mean, g_mean, b_mean: Channel means (0-1)
            - edge_density: Fraction of strong Sobel edges
            - entropy: Grayscale histogram entropy (bits)
            - highfreq_std: Std of high-frequency component

    Raises:
        ValueError: If label_index is empty.
    """
    if not label_index:
        raise ValueError("label_index is empty")

    rng = np.random.default_rng(seed)
    rows = []

    for cls_id, idxs in tqdm(label_index.items(), desc="Computing metrics"):
        if len(idxs) == 0:
            continue

        take = rng.choice(
            idxs,
            size=min(sample_per_class, len(idxs)),
            replace=False,
        )

        for idx in take:
            idx = int(idx)
            try:
                img = ds["images"][idx].numpy()
                if img.ndim != 3 or img.shape[2] != 3:
                    logger.warning(f"Skipping idx={idx}: unexpected shape {img.shape}")
                    continue

                if img.dtype != np.uint8:
                    img_float = img.astype(np.float32)
                    if img_float.max() <= 1.0:
                        img_float = np.clip(img_float, 0.0, 1.0)
                        img_uint8 = (img_float * 255.0).round().astype(np.uint8)
                    else:
                        img_float = np.clip(img_float, 0.0, 255.0)
                        img_uint8 = img_float.astype(np.uint8)
                else:
                    img_uint8 = img

                h, w = img_uint8.shape[:2]
                aspect_ratio = w / h

                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
                gray_f = gray.astype(np.float32)

                brightness = gray_f.mean() / 255.0
                contrast = gray_f.std() / 255.0

                lap = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=3)
                blur = lap.var()

                img_norm = img_uint8.astype(np.float32) / 255.0
                r_mean = img_norm[:, :, 0].mean()
                g_mean = img_norm[:, :, 1].mean()
                b_mean = img_norm[:, :, 2].mean()

                sx = ndimage.sobel(gray_f, axis=0)
                sy = ndimage.sobel(gray_f, axis=1)
                edge_mag = np.hypot(sx, sy)

                if np.all(edge_mag == 0):
                    edge_density = 0.0
                else:
                    thresh = np.percentile(edge_mag, edge_percentile)
                    if thresh <= 0:
                        edge_density = 0.0
                    else:
                        edge_density = (edge_mag > thresh).mean()

                hist, _ = np.histogram(
                    gray.ravel(),
                    bins=256,
                    range=(0, 255),
                    density=True,
                )
                hist_nonzero = hist[hist > 0]
                entropy = (
                    -(hist_nonzero * np.log2(hist_nonzero)).sum()
                    if hist_nonzero.size > 0
                    else 0.0
                )

                smoothed = ndimage.gaussian_filter(gray_f, sigma=1.5)
                highfreq = gray_f - smoothed
                highfreq_std = highfreq.std()

                rows.append(
                    {
                        "idx": idx,
                        "class_id": cls_id,
                        "h": h,
                        "w": w,
                        "aspect_ratio": aspect_ratio,
                        "brightness": brightness,
                        "contrast": contrast,
                        "blur": blur,
                        "r_mean": r_mean,
                        "g_mean": g_mean,
                        "b_mean": b_mean,
                        "edge_density": edge_density,
                        "entropy": entropy,
                        "highfreq_std": highfreq_std,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed computing metrics for idx={idx}: {e}")
                continue

    return pd.DataFrame(rows)


def add_zscores_by_split(
    df: pd.DataFrame, metric_cols: List[str], z_thresh: float = 2.5
) -> pd.DataFrame:
    """
    Add z-scores and outlier flags computed per-class within each split.

    For each metric, adds two new columns:
        - {metric}_z: Z-score relative to class mean within split
        - {metric}_is_outlier: Boolean flag if |z-score| > z_thresh

    Args:
        df: DataFrame with 'split' and 'class_id' columns plus metric columns.
        metric_cols: List of column names to compute z-scores for.
        z_thresh: Threshold for outlier detection (default 2.5).

    Returns:
        pd.DataFrame: Copy of input DataFrame with added z-score columns.
    """
    df = df.copy()
    for m in metric_cols:
        z_col = f"{m}_z"
        out_col = f"{m}_is_outlier"
        df[z_col] = df.groupby(["split", "class_id"])[m].transform(
            lambda s: (
                stats.zscore(s, nan_policy="omit") if len(s) >= 3 and s.std() > 0 else 0
            )
        )
        df[z_col] = df[z_col].fillna(0)
        df[out_col] = df[z_col].abs() > z_thresh
    return df


# ============================================================
# METRIC VISUALIZATION
# ============================================================


def plot_metric_box_and_hist_seaborn(
    df: pd.DataFrame,
    metric: str,
    train_color: str,
    val_color: str,
    metric_label: str | None = None,
    bins: int = 30,
    thresholds: Tuple[float, float] | None = None,
) -> None:
    """
    Plot horizontal boxplot and histogram comparing Train vs Val for a metric.

    Creates a 2-row figure:
        1. Horizontal boxplot by split
        2. Histogram with percentage on y-axis and median lines

    Args:
        df: DataFrame with 'split' column and the metric column.
        metric: Name of the metric column to visualize.
        train_color: Color for training data.
        val_color: Color for validation data.
        metric_label: Display name for the metric (defaults to column name).
        bins: Number of histogram bins.
        thresholds: Optional (low, high) tuple to draw vertical threshold lines.

    Returns:
        None
    """

    if metric_label is None:
        metric_label = metric

    train_vals = df.loc[df["split"] == "train", metric].dropna()
    val_vals = df.loc[df["split"] == "val", metric].dropna()

    tidy = df[["split", metric]].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))

    sns.boxplot(
        data=tidy,
        x=metric,
        y="split",
        hue="split",
        orient="h",
        palette={"train": train_color, "val": val_color},
        linewidth=2.0,
        fliersize=2,
        width=0.6,
        ax=axes[0],
    )

    axes[0].legend([], [], frameon=False)

    if thresholds is not None:
        low, high = thresholds
        axes[0].axvline(low, color="orange", linestyle=":", alpha=0.8)
        axes[0].axvline(high, color="orange", linestyle=":", alpha=0.8)

    axes[0].set_title(f"{metric_label}: Train vs Val (Horizontal Boxplot)", fontsize=13)
    axes[0].set_xlabel(metric_label)
    axes[0].set_ylabel("")
    axes[0].grid(True, axis="x", alpha=0.3)

    train_weights = np.ones_like(train_vals) / len(train_vals) * 100
    val_weights = np.ones_like(val_vals) / len(val_vals) * 100

    axes[1].hist(
        train_vals,
        bins=bins,
        weights=train_weights,
        alpha=0.6,
        color=train_color,
        label="Train",
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].hist(
        val_vals,
        bins=bins,
        weights=val_weights,
        alpha=0.6,
        color=val_color,
        label="Val",
        edgecolor="black",
        linewidth=0.6,
    )

    axes[1].axvline(
        train_vals.median(),
        color=train_color,
        linestyle="--",
        lw=2,
        alpha=0.9,
        label=f"Train med = {train_vals.median():.3f}",
    )
    axes[1].axvline(
        val_vals.median(),
        color=val_color,
        linestyle="--",
        lw=2,
        alpha=0.9,
        label=f"Val med = {val_vals.median():.3f}",
    )

    if thresholds is not None:
        low, high = thresholds
        axes[1].axvline(low, color="orange", linestyle=":", lw=1.5, alpha=0.8)
        axes[1].axvline(high, color="orange", linestyle=":", lw=1.5, alpha=0.8)

    axes[1].set_title(f"{metric_label} Distribution (% per bin)", fontsize=13)
    axes[1].set_xlabel(metric_label)
    axes[1].set_ylabel("Percentage of images (%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def show_metric_examples_low_mid_high(
    df: pd.DataFrame,
    metric: str,
    ds_train: DeepLakeDataset,
    ds_val: DeepLakeDataset,
    id_to_name: Dict[int, str],
    k: int = 3,
    title: str | None = None,
    metric_label: str | None = None,
) -> None:
    """
    Show k lowest, k median, and k highest images for a given metric.

    Creates a 3-row grid visualization:
        - Row 1: k images with lowest metric values
        - Row 2: k images closest to the median
        - Row 3: k images with highest metric values

    Args:
        df: DataFrame with columns ['idx', 'split', 'class_id', metric].
        metric: Metric column name (e.g., 'brightness').
        ds_train: DeepLake training dataset with 'images' tensor.
        ds_val: DeepLake validation dataset with 'images' tensor.
        id_to_name: Mapping from class_id to class name.
        k: Number of images in each group (rows).
        title: Figure title (uses default if None).
        metric_label: Display name for the metric (defaults to column name).

    Returns:
        None
    """
    if metric_label is None:
        metric_label = metric

    df_valid = df.dropna(subset=[metric]).copy()

    low_df = df_valid.nsmallest(k, metric)

    median_val = df_valid[metric].median()
    df_valid["dist_to_median"] = (df_valid[metric] - median_val).abs()
    mid_df = df_valid.nsmallest(k, "dist_to_median")

    high_df = df_valid.nlargest(k, metric)

    groups = [low_df, mid_df, high_df]
    row_labels = [
        f"LOW ({metric} min)",
        f"MID (median ≈ {median_val:.2f})",
        f"HIGH ({metric} max)",
    ]

    fig, axes = plt.subplots(3, k, figsize=(4 * k, 10))

    if k == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for row_idx, (group_df, row_label) in enumerate(zip(groups, row_labels)):
        for col_idx in range(k):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(group_df):
                ax.axis("off")
                continue

            row = group_df.iloc[col_idx]
            ds = ds_train if row["split"] == "train" else ds_val
            img = ds["images"][int(row["idx"])].numpy()

            ax.imshow(img)
            ax.axis("off")

            cname = id_to_name.get(row["class_id"], str(row["class_id"]))[:18]
            split_label = "T" if row["split"] == "train" else "V"

            ax.set_title(
                f"{cname}\n{metric_label}={row[metric]:.3f} ({split_label})",
                fontsize=8,
            )

        axes[row_idx, 0].set_ylabel(
            row_label,
            fontsize=11,
            fontweight="bold",
            rotation=0,
            labelpad=50,
            va="center",
        )

    if title is None:
        title = f"{metric_label}: Example Images (Low / Median / High)"

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_top_bottom_classes_by_split(
    df: pd.DataFrame,
    metric: str,
    id_to_name: Dict[int, str],
    n_classes: int = 3,
    metric_label: str | None = None,
) -> Dict[str, List[int]]:
    """
    Show top and bottom classes by metric for train and validation splits.

    Creates horizontal barplots showing classes with highest and lowest
    mean metric values, separately for each split.

    Args:
        df: DataFrame with 'split', 'class_id', and metric columns.
        metric: Metric column name to analyze.
        id_to_name: Mapping from class_id to human-readable name.
        n_classes: Number of top/bottom classes to show.
        metric_label: Display name for the metric (defaults to column name).

    Returns:
        Dict[str, List[int]]: Dictionary containing:
            - 'train_bottom_ids': Class IDs with lowest metric in training
            - 'train_top_ids': Class IDs with highest metric in training
            - 'val_bottom_ids': Class IDs with lowest metric in validation
            - 'val_top_ids': Class IDs with highest metric in validation
    """
    if metric_label is None:
        metric_label = metric

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_means = train_df.groupby("class_id")[metric].mean().sort_values()
    val_means = val_df.groupby("class_id")[metric].mean().sort_values()

    train_bottom = train_means.head(n_classes)
    train_top = train_means.tail(n_classes)

    val_bottom = val_means.head(n_classes)
    val_top = val_means.tail(n_classes)

    def to_tidy(series, split, group):
        out = series.reset_index()
        out["class_name"] = out["class_id"].map(lambda x: id_to_name.get(x, f"ID_{x}"))
        out["split"] = split
        out["group"] = group
        return out

    tidy = pd.concat(
        [
            to_tidy(train_bottom, "Train", "Bottom"),
            to_tidy(train_top, "Train", "Top"),
            to_tidy(val_bottom, "Val", "Bottom"),
            to_tidy(val_top, "Val", "Top"),
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.barplot(
        data=tidy[tidy["split"] == "Train"],
        x=metric,
        y="class_name",
        hue="group",
        palette={"Bottom": "#4C72B0", "Top": "#DD8452"},
        orient="h",
        ax=axes[0],
    )
    axes[0].set_title(f"Train: Top & Bottom {n_classes} Classes")
    axes[0].set_xlabel(metric_label)
    axes[0].set_ylabel("Class")
    axes[0].grid(True, axis="x", alpha=0.3)

    sns.barplot(
        data=tidy[tidy["split"] == "Val"],
        x=metric,
        y="class_name",
        hue="group",
        palette={"Bottom": "#4C72B0", "Top": "#DD8452"},
        orient="h",
        ax=axes[1],
    )
    axes[1].set_title(f"Val: Top & Bottom {n_classes} Classes")
    axes[1].set_xlabel(metric_label)
    axes[1].set_ylabel("Class")
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.suptitle(
        f"{metric_label}: Lowest & Highest Classes (Train vs Val)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return {
        "train_bottom_ids": list(train_bottom.index),
        "train_top_ids": list(train_top.index),
        "val_bottom_ids": list(val_bottom.index),
        "val_top_ids": list(val_top.index),
    }


def show_class_examples_for_split(
    df: pd.DataFrame,
    metric: str,
    ds_train: DeepLakeDataset,
    ds_val: DeepLakeDataset,
    id_to_name: Dict[int, str],
    class_ids_train_bottom: List[int],
    class_ids_train_top: List[int],
    class_ids_val_bottom: List[int],
    class_ids_val_top: List[int],
    add_spacing: bool = True,
    title: str | None = None,
) -> None:
    """
    Show example images for top/bottom classes in train and validation splits.

    Creates a 4-row grid showing:
        - Train bottom classes (lowest metric)
        - Train top classes (highest metric)
        - Val bottom classes (lowest metric)
        - Val top classes (highest metric)

    Args:
        df: DataFrame with 'split', 'class_id', 'idx', and metric columns.
        metric: Metric column name for display.
        ds_train: DeepLake training dataset with 'images' tensor.
        ds_val: DeepLake validation dataset with 'images' tensor.
        id_to_name: Mapping from class_id to human-readable name.
        class_ids_train_bottom: List of class IDs for train bottom group.
        class_ids_train_top: List of class IDs for train top group.
        class_ids_val_bottom: List of class IDs for val bottom group.
        class_ids_val_top: List of class IDs for val top group.
        add_spacing: Whether to add spacer columns for readability.
        title: Figure title (uses default if None).

    Returns:
        None
    """

    groups = [
        ("TRAIN — Lowest Classes", class_ids_train_bottom, "train"),
        ("TRAIN — Highest Classes", class_ids_train_top, "train"),
        ("VAL — Lowest Classes", class_ids_val_bottom, "val"),
        ("VAL — Highest Classes", class_ids_val_top, "val"),
    ]

    max_classes = max(len(g[1]) for g in groups)
    if add_spacing:
        n_cols = max_classes + 1
    else:
        n_cols = max_classes

    fig, axes = plt.subplots(len(groups), n_cols, figsize=(4 * n_cols, 10))

    if n_cols == 1:
        axes = axes.reshape(len(groups), 1)

    def get_example(split_df, ds, class_id):
        rows = split_df[split_df["class_id"] == class_id]
        if len(rows) == 0:
            return None, None
        row = rows.iloc[0]
        return ds["images"][int(row["idx"])].numpy(), row[metric]

    for row_idx, (label, class_ids, split) in enumerate(groups):
        split_df = df[df["split"] == split]
        ds = ds_train if split == "train" else ds_val

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if add_spacing and col_idx == max_classes:
                ax.axis("off")
                continue

            if col_idx >= len(class_ids):
                ax.axis("off")
                continue

            cid = class_ids[col_idx]
            cname = id_to_name.get(cid, f"ID_{cid}")

            img, val = get_example(split_df, ds, cid)
            if img is None:
                ax.axis("off")
                continue

            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{cname[:20]}\n{metric}={val:.3f}", fontsize=8)

        axes[row_idx, 0].annotate(
            label,
            xy=(-0.15, 0.5),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )

    if title is None:
        title = f"{metric.upper()} – Class-Level Examples (Train vs Val)"

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_brightness_vs_contrast(
    df: pd.DataFrame,
    train_color: str,
    val_color: str,
    alpha: float = 0.6,
    s: float = 40,
) -> None:
    """
    Create a scatter plot of brightness vs contrast, colored by split.

    Args:
        df: DataFrame with 'brightness', 'contrast', and 'split' columns.
        train_color: Color for training data points.
        val_color: Color for validation data points.
        alpha: Transparency of scatter points (0-1).
        s: Size of scatter points.

    Returns:
        None
    """

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=df,
        x="brightness",
        y="contrast",
        hue="split",
        palette={"train": train_color, "val": val_color},
        alpha=alpha,
        s=s,
        edgecolor="none",
    )

    plt.title("Brightness vs Contrast", fontsize=14, fontweight="bold")
    plt.xlabel("Brightness (0–1)")
    plt.ylabel("Contrast (Std of grayscale)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Split")
    plt.tight_layout()
    plt.show()


# ============================================================
# FEATURE EXTRACTION & EMBEDDINGS
# ============================================================


def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Priority order: CUDA GPU > Apple MPS > CPU.

    Returns:
        torch.device: The selected device for tensor operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class FeatureExtractor:
    """
    Wrapper around a pretrained ResNet50 used as a feature extractor.

    Loads ResNet50 pretrained on ImageNet and removes the final classification
    head so output is a 2048-dimensional embedding. Provides batched extraction
    helpers for efficient processing of large datasets.

    Attributes:
        device: Torch device for computation (cuda, mps, or cpu).
        model: ResNet50 model with classification head replaced by Identity.
        preprocess: ImageNet preprocessing transforms.
        embedding_dim: Dimension of output embeddings (2048).
    """

    def __init__(self, device: torch.device) -> None:
        """
        Initialize the feature extractor with a pretrained ResNet50.

        Args:
            device: Torch device to run the model on.
        """
        self.device = device

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)

        model.fc = nn.Identity()

        self.model = model.to(device).eval()
        self.preprocess = weights.transforms()
        self.embedding_dim = 2048

    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for a batch of images.

        Args:
            images: List of numpy arrays with shape (H, W, C), dtype uint8.

        Returns:
            np.ndarray: Embeddings with shape (batch_size, embedding_dim).
        """
        processed = []
        for img in images:
            arr = np.asarray(img)
            if arr.dtype != np.uint8:
                max_val = float(arr.max()) if arr.size > 0 else 1.0
                if max_val <= 1.0 + 1e-6:
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(arr)
            processed.append(self.preprocess(pil_img))
        tensors = torch.stack(processed).to(self.device)

        feats = self.model(tensors)
        return feats.cpu().numpy()

    @torch.no_grad()
    def extract_from_dataset(
        self,
        ds: DeepLakeDataset,
        indices: np.ndarray,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Extract embeddings from dataset indices in batches.

        Args:
            ds: DeepLake dataset where ds[i]["images"].numpy() returns an image.
            indices: 1D array of dataset indices to extract embeddings from.
            batch_size: Number of images to process per batch.

        Returns:
            np.ndarray: Embeddings with shape (N, embedding_dim) where N is
                the number of indices.
        """
        all_embs = []
        indices = np.array(indices, dtype=int)

        for start in tqdm(
            range(0, len(indices), batch_size), desc="Extracting features"
        ):
            batch_idx = indices[start : start + batch_size]
            batch_imgs = [ds[int(i)]["images"].numpy() for i in batch_idx]
            emb = self.extract_batch(batch_imgs)
            all_embs.append(emb)

        return np.vstack(all_embs)


# ---------------------------------------
# Extract embeddings & class prototypes
# ---------------------------------------
def extract_class_embeddings(
    ds: DeepLakeDataset,
    label_index: Dict[int, np.ndarray],
    extractor: FeatureExtractor,
    samples_per_class: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ResNet50 embeddings and compute class prototypes.

    Samples images from each class, extracts feature embeddings, and
    computes a class prototype (mean embedding) for each class.

    Args:
        ds: DeepLake dataset with 'images' tensor.
        label_index: Dictionary mapping class_id to array of dataset indices.
        extractor: FeatureExtractor instance for embedding extraction.
        samples_per_class: Number of images to sample per class.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple containing:
            - embeddings: (N, D) array of image embeddings
            - labels: (N,) array of class_ids for each embedding
            - prototypes: (C, D) array of class prototype embeddings
            - proto_class_ids: (C,) array of class_ids for each prototype
    """
    rng = np.random.default_rng(seed)

    all_indices = []
    all_labels = []

    for class_id in sorted(label_index.keys()):
        idxs = np.array(label_index[class_id])
        if len(idxs) == 0:
            continue

        n_samples = min(samples_per_class, len(idxs))
        sampled = rng.choice(idxs, size=n_samples, replace=False)

        all_indices.extend(sampled.tolist())
        all_labels.extend([class_id] * n_samples)

    all_indices = np.array(all_indices, dtype=int)
    all_labels = np.array(all_labels, dtype=int)

    print(f"Total sampled images: {len(all_indices)}")
    print(f"Unique classes (non-empty): {len(np.unique(all_labels))}")

    embeddings = extractor.extract_from_dataset(ds, all_indices, batch_size=64)
    print("Embeddings shape:", embeddings.shape)

    prototypes = []
    proto_class_ids = []
    for class_id in sorted(np.unique(all_labels)):
        mask = all_labels == class_id
        class_embs = embeddings[mask]
        proto = class_embs.mean(axis=0)
        prototypes.append(proto)
        proto_class_ids.append(class_id)

    prototypes = np.vstack(prototypes)
    proto_class_ids = np.array(proto_class_ids, dtype=int)
    print("Prototypes shape:", prototypes.shape)

    return embeddings, all_labels, prototypes, proto_class_ids


# ============================================================
# CLASS SIMILARITY ANALYSIS
# ============================================================


def compute_class_similarity(
    prototypes: np.ndarray, class_ids: np.ndarray
) -> pd.DataFrame:
    """
    Compute pairwise cosine similarity between class prototypes.

    Args:
        prototypes: (C, D) array of class prototype embeddings.
        class_ids: (C,) array of class IDs corresponding to each prototype.

    Returns:
        pd.DataFrame: Similarity matrix with shape (C, C), where index and
            columns are class_ids. Values are cosine similarities in [-1, 1].
    """
    prototypes = np.asarray(prototypes, dtype=np.float32)
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
    prot_norm = prototypes / norms

    sim_matrix = prot_norm @ prot_norm.T

    return pd.DataFrame(sim_matrix, index=class_ids, columns=class_ids)


def get_most_similar_pairs(
    similarity_df: pd.DataFrame, n_pairs: int = 20
) -> pd.DataFrame:
    """
    Extract the top n most similar class pairs from a similarity matrix.

    Excludes the diagonal (self-similarity) and considers only the upper
    triangle to avoid duplicate pairs.

    Args:
        similarity_df: DataFrame with pairwise class similarities.
        n_pairs: Number of top similar pairs to return.

    Returns:
        pd.DataFrame: DataFrame with columns ['class_1', 'class_2', 'similarity'],
            sorted by similarity in descending order.
    """
    pairs = []
    class_ids = similarity_df.index.to_list()

    for i in range(len(class_ids)):
        for j in range(i + 1, len(class_ids)):
            pairs.append(
                {
                    "class_1": class_ids[i],
                    "class_2": class_ids[j],
                    "similarity": similarity_df.iloc[i, j],
                }
            )

    pairs_df = pd.DataFrame(pairs)
    return pairs_df.sort_values("similarity", ascending=False).head(n_pairs)


def show_class_pair_examples(
    ds: DeepLakeDataset,
    label_index: Dict[int, np.ndarray],
    cls_a: int,
    cls_b: int,
    id_to_name: Dict[int, str],
    similarity: float | None = None,
    n_per_class: int = 3,
    seed: int = 42,
) -> None:
    """
    Display example images from two classes side by side.

    Useful for visually inspecting why two classes appear similar in
    embedding space.

    Args:
        ds: DeepLake dataset with 'images' tensor.
        label_index: Dictionary mapping class_id to array of dataset indices.
        cls_a: First class ID to display.
        cls_b: Second class ID to display.
        id_to_name: Mapping from class_id to human-readable name.
        similarity: Optional similarity score to display in title.
        n_per_class: Number of examples to show per class.
        seed: Random seed for reproducible sampling.

    Returns:
        None
    """
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6))

    for row, cls in enumerate([cls_a, cls_b]):
        idxs = np.array(label_index[cls])
        n_show = min(n_per_class, len(idxs))
        chosen = rng.choice(idxs, size=n_show, replace=False)

        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < n_show:
                img = ds[int(chosen[col])]["images"].numpy()
                name = id_to_name.get(cls, str(cls))
                ax.imshow(img)
                title_prefix = "A" if row == 0 else "B"
                ax.set_title(f"{title_prefix}: {cls}\n{name}", fontsize=8)

    if similarity is not None:
        plt.suptitle(
            f"Class pair: {cls_a} ↔ {cls_b}  |  Similarity: {similarity:.4f}",
            fontsize=12,
        )
    else:
        plt.suptitle(f"Class pair: {cls_a} ↔ {cls_b}", fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# IMAGE PREPROCESSING TRANSFORMS
# ============================================================


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances local contrast by applying histogram equalization on the L
    channel in LAB color space. Particularly effective for dark images
    and images with low contrast.

    Args:
        img: RGB image as uint8 array with shape (H, W, 3).
        clip_limit: Contrast limit for adaptive histogram equalization.
            Higher values allow more contrast.
        tile_grid_size: Number of tiles in (rows, cols) for the algorithm.

    Returns:
        np.ndarray: Enhanced RGB image as uint8 array with same shape.
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    out = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Apply gamma correction for smooth brightness adjustment.

    Gamma correction modifies pixel intensity using the formula:
    output = input^(1/gamma) * 255

    Args:
        img: RGB image as uint8 array.
        gamma: Gamma factor. Values > 1 brighten the image,
            values < 1 darken it.

    Returns:
        np.ndarray: Gamma-corrected image as uint8 array.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, table)


def apply_sharpening(
    img: np.ndarray,
    amount: float = 1.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply unsharp masking to sharpen an image.

    Enhances edges by subtracting a blurred version of the image
    from the original. Particularly useful for blurry images of
    fast-moving subjects.

    Args:
        img: RGB image as uint8 array.
        amount: Strength of the sharpening effect. Higher values
            produce more pronounced edges.
        sigma: Standard deviation of the Gaussian blur kernel.

    Returns:
        np.ndarray: Sharpened image as uint8 array.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_bbox_crop(
    img: np.ndarray,
    ds: DeepLakeDataset,
    idx: int,
    padding_ratio: float = 0.1,
) -> np.ndarray:
    """
    Crop image to bounding box with optional padding.

    Extracts the region defined by the bounding box with additional
    padding around the edges. Useful for removing background context
    and focusing on the subject.

    Args:
        img: Input RGB image as uint8 array with shape (H, W, C).
        ds: DeepLake dataset containing bounding box annotations.
        idx: Index of the sample in the dataset.
        padding_ratio: Fraction of bbox dimensions to add as padding
            on each side.

    Returns:
        np.ndarray: Cropped image as uint8 array. Returns original image
            if bounding box is missing or invalid.
    """
    bbox = resolve_bbox_xywh_or_xyxy(ds, int(idx))
    if bbox is None:
        return img

    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    box_w, box_h = x2 - x1, y2 - y1
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return img

    return img[y1:y2, x1:x2]


# ============================================================
# BUILD TEST CASES FROM EDA METRICS
# ============================================================


def build_test_cases(
    metrics_df: pd.DataFrame, id_to_name: Dict[int, str], n_samples: int = 3
) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Build test cases from quality metrics for preprocessing visualization.

    Identifies images with specific quality characteristics (dark, low
    contrast, blurry, complex background, normal) based on metric
    percentiles.

    Args:
        metrics_df: DataFrame with 'split', 'idx', 'class_id', and metric
            columns ('brightness', 'contrast', 'blur', 'edge_density').
        id_to_name: Mapping from class_id to human-readable name.
        n_samples: Number of samples per category.

    Returns:
        Dict[str, List[Tuple[int, int, str]]]: Dictionary mapping category
            name to list of (idx, class_id, class_name) tuples.
            Categories: 'dark', 'low_contrast', 'blurry',
            'complex_background', 'normal'.
    """
    train_df = metrics_df[metrics_df["split"] == "train"].copy()

    test_cases = {}

    brightness_thresh = train_df["brightness"].quantile(0.05)
    dark_samples = train_df[train_df["brightness"] < brightness_thresh].head(n_samples)
    test_cases["dark"] = [
        (
            int(row["idx"]),
            int(row["class_id"]),
            id_to_name.get(int(row["class_id"]), "Unknown"),
        )
        for _, row in dark_samples.iterrows()
    ]

    contrast_thresh = train_df["contrast"].quantile(0.05)
    low_contrast_samples = train_df[train_df["contrast"] < contrast_thresh].head(
        n_samples
    )
    test_cases["low_contrast"] = [
        (
            int(row["idx"]),
            int(row["class_id"]),
            id_to_name.get(int(row["class_id"]), "Unknown"),
        )
        for _, row in low_contrast_samples.iterrows()
    ]

    blur_thresh = train_df["blur"].quantile(0.05)
    blurry_samples = train_df[train_df["blur"] < blur_thresh].head(n_samples)
    test_cases["blurry"] = [
        (
            int(row["idx"]),
            int(row["class_id"]),
            id_to_name.get(int(row["class_id"]), "Unknown"),
        )
        for _, row in blurry_samples.iterrows()
    ]

    edge_thresh = train_df["edge_density"].quantile(0.95)
    complex_bg_samples = train_df[train_df["edge_density"] > edge_thresh].head(
        n_samples
    )
    test_cases["complex_background"] = [
        (
            int(row["idx"]),
            int(row["class_id"]),
            id_to_name.get(int(row["class_id"]), "Unknown"),
        )
        for _, row in complex_bg_samples.iterrows()
    ]

    median_brightness = train_df["brightness"].median()
    median_contrast = train_df["contrast"].median()
    normal_samples = train_df[
        train_df["brightness"].between(median_brightness - 0.1, median_brightness + 0.1)
        & train_df["contrast"].between(median_contrast - 0.05, median_contrast + 0.05)
    ].head(n_samples)
    test_cases["normal"] = [
        (
            int(row["idx"]),
            int(row["class_id"]),
            id_to_name.get(int(row["class_id"]), "Unknown"),
        )
        for _, row in normal_samples.iterrows()
    ]

    return test_cases


# ============================================================
# VISUALIZE PREPROCESSING TRANSFORMS
# ============================================================


def visualize_preprocessing(
    ds: DeepLakeDataset,
    test_cases: Dict[str, List[Tuple[int, int, str]]],
    transforms_dict: Dict[
        str, Callable[[np.ndarray, DeepLakeDataset, int], np.ndarray]
    ],
    categories: List[str] | None = None,
    figsize_scale: float = 2.5,
) -> None:
    """
    Visualize preprocessing transforms on sample images by category.

    Creates a grid for each category where:
        - Rows represent sample images
        - Columns represent different preprocessing transforms

    Args:
        ds: DeepLake dataset with 'images' tensor.
        test_cases: Dictionary mapping category names to lists of
            (idx, class_id, class_name) tuples.
        transforms_dict: Dictionary mapping transform names to functions.
            Each function should accept (img, ds, idx) and return an image.
        categories: List of categories to visualize. If None, uses all
            categories in test_cases.
        figsize_scale: Scale factor for figure size.

    Returns:
        None
    """
    if categories is None:
        categories = list(test_cases.keys())

    transform_names = list(transforms_dict.keys())
    n_transforms = len(transform_names)

    for category in categories:
        if category not in test_cases or len(test_cases[category]) == 0:
            print(f"No samples for category: {category}")
            continue

        samples = test_cases[category]
        n_samples = len(samples)

        fig, axes = plt.subplots(
            n_samples,
            n_transforms,
            figsize=(figsize_scale * n_transforms, figsize_scale * n_samples),
        )
        if n_samples == 1:
            axes = np.expand_dims(axes, 0)

        fig.suptitle(
            f"Preprocessing: {category.upper().replace('_', ' ')}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        for row, (idx, class_id, class_name) in enumerate(samples):
            img = ds[int(idx)]["images"].numpy()

            for col, t_name in enumerate(transform_names):
                ax = axes[row, col]
                try:
                    transformed = transforms_dict[t_name](img.copy(), ds, int(idx))
                    ax.imshow(transformed)
                    if row == 0:
                        ax.set_title(t_name.replace("_", "\n"), fontsize=9)
                    if col == 0:
                        ax.set_ylabel(f"{class_name[:20]}...\nidx={idx}", fontsize=8)
                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error:\n{str(e)[:30]}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        transform=ax.transAxes,
                    )
                ax.axis("off")

        plt.tight_layout()
        plt.show()


# ============================================================
# AUGMENTATION PIPELINES
# ============================================================


def get_augmentation_pipelines(
    img_size: int = 224, for_viz: bool = False
) -> Dict[str, A.Compose]:
    """
    Define augmentation strategies based on EDA findings.

    Creates multiple augmentation pipelines optimized for different
    scenarios identified during exploratory data analysis:
        - baseline: Aspect-preserving resize + pad (no augmentation)
        - standard: Balanced augmentation for general training
        - geometric_heavy: For similar species pairs (pose/shape variation)
        - color_conservative: For distinctive plumage species
        - background_robust: Reduce background dependency
        - quality_adaptive: For dark/low-contrast images

    Args:
        img_size: Target image size (creates square img_size x img_size).
        for_viz: If True, omit Normalize/ToTensorV2 so images can be
            displayed directly with matplotlib.

    Returns:
        Dict[str, A.Compose]: Dictionary mapping pipeline names to
            Albumentations Compose objects.
    """
    norm_transforms = (
        []
        if for_viz
        else [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    pipelines = {
        "baseline": A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
            ]
            + norm_transforms
        ),
        "standard": A.Compose(
            [
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(1.0, 1.0),
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3,
                ),
            ]
            + norm_transforms
        ),
        "geometric_heavy": A.Compose(
            [
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.6, 1.0),
                    ratio=(1.0, 1.0),
                ),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    rotate=(-20, 20),
                    shear=(-10, 10),
                    scale=(0.9, 1.1),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.Perspective(
                    scale=(0.02, 0.06), border_mode=cv2.BORDER_REFLECT_101, p=0.3
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3,
                ),
            ]
            + norm_transforms
        ),
        "color_conservative": A.Compose(
            [
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.85, 1.0),
                    ratio=(1.0, 1.0),
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3,
                ),
            ]
            + norm_transforms
        ),
        "background_robust": A.Compose(
            [
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.5, 1.0),
                    ratio=(1.0, 1.0),
                ),
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(img_size // 8, img_size // 4),
                    hole_width_range=(img_size // 8, img_size // 4),
                    p=0.3,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.GaussNoise(std_range=(0.0, 0.1), p=0.2),
            ]
            + norm_transforms
        ),
        "quality_adaptive": A.Compose(
            [
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.3,
                            contrast_limit=0.3,
                            p=1.0,
                        ),
                        A.RandomGamma(
                            gamma_limit=(80, 120),
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size,
                    min_width=img_size,
                    border_mode=cv2.BORDER_REFLECT_101,
                ),
                A.HorizontalFlip(p=0.5),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
            ]
            + norm_transforms
        ),
    }

    return pipelines


IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_bbox_crop_transform(img_size: int = 224, for_viz: bool = False) -> A.Compose:
    """
    Create an augmentation pipeline for use after bounding box cropping.

    Designed to be applied after apply_bbox_crop(). Includes:
        1. LongestMaxSize to preserve aspect ratio
        2. PadIfNeeded with reflection (no black borders)
        3. Light geometric and color augmentations

    Args:
        img_size: Target image size (creates square img_size x img_size).
        for_viz: If True, omit Normalize/ToTensorV2 so images can be
            displayed directly with matplotlib.

    Returns:
        A.Compose: Albumentations pipeline ready for bbox-cropped images.
    """
    norm_transforms = (
        []
        if for_viz
        else [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                rotate=(-15, 15),
                shear=(-5, 5),
                scale=(0.95, 1.05),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=8,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=0.3,
            ),
        ]
        + norm_transforms
    )


# ============================================================
# VISUALIZE AUGMENTATION PIPELINES
# ============================================================


def visualize_augmentations(
    ds: DeepLakeDataset,
    indices: List[int],
    pipeline_fns: Dict[str, Callable[[np.ndarray, DeepLakeDataset, int], np.ndarray]],
    id_to_name: Dict[int, str] | None = None,
    figsize_scale: float = 2.5,
) -> None:
    """
    Visualize augmentation pipelines side-by-side for multiple images.

    Creates a grid where:
        - Rows represent different dataset indices
        - Column 0 shows the original image
        - Columns 1+ show results from each pipeline

    Args:
        ds: DeepLake dataset with 'images' and 'labels' tensors.
        indices: List of dataset indices to visualize.
        pipeline_fns: Dictionary mapping pipeline names to functions.
            Each function should accept (img, ds, idx) and return an image.
        id_to_name: Optional mapping from class_id to human-readable name.
        figsize_scale: Scale factor for figure size.

    Returns:
        None
    """
    if len(indices) == 0:
        print("No indices provided for visualization.")
        return

    indices = [int(i) for i in indices]
    pipeline_names = list(pipeline_fns.keys())
    n_pipelines = len(pipeline_names)
    n_images = len(indices)

    fig, axes = plt.subplots(
        n_images,
        n_pipelines + 1,
        figsize=(figsize_scale * (n_pipelines + 1), figsize_scale * n_images),
    )
    if n_images == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "Augmentation pipelines by image (rows = images, cols = pipelines)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    for row, idx in enumerate(indices):
        idx = int(idx)
        img = ds[int(idx)]["images"].numpy()

        class_name = ""
        try:
            label_arr = ds[int(idx)]["labels"].numpy()
            label_id = int(label_arr.flat[0])
            if id_to_name is not None:
                class_name = id_to_name.get(label_id, f"Class {label_id}")[:25]
            else:
                class_name = f"Class {label_id}"
        except Exception:
            class_name = ""

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(
            f"idx={idx}\n{class_name}",
            fontsize=8,
        )
        if row == 0:
            axes[row, 0].set_title("Original", fontsize=9)
        axes[row, 0].axis("off")

        for col, p_name in enumerate(pipeline_names, start=1):
            ax = axes[row, col]
            fn = pipeline_fns[p_name]
            try:
                aug = fn(img.copy(), ds, idx)
                if isinstance(aug, dict):
                    aug = aug.get("image", aug)
                if hasattr(aug, "dtype") and aug.dtype != np.uint8:
                    aug_vis = np.clip(aug, 0.0, 1.0)
                else:
                    aug_vis = aug
                ax.imshow(aug_vis)
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"Error\n{str(e)[:20]}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    transform=ax.transAxes,
                )
            if row == 0:
                ax.set_title(p_name.replace("_", "\n"), fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================
# COLOR BIAS VISUALIZATION
# ============================================================


def show_color_bias_class_examples(
    df: pd.DataFrame,
    ds_train: DeepLakeDataset,
    ds_val: DeepLakeDataset,
    id_to_name: Dict[int, str],
    color_bias_results: Dict[str, Dict[str, List[int]]],
    n_examples: int = 2,
    seed: int = 42,
) -> None:
    """
    Show example images for top color-biased classes.

    Args:
        df: DataFrame with columns ['split', 'class_id', 'idx', 'r_bias', 'g_bias', 'b_bias']
        ds_train: DeepLake training dataset
        ds_val: DeepLake validation dataset
        id_to_name: Mapping from class ID to class name
        color_bias_results: Dict with keys 'r_bias', 'g_bias', 'b_bias', each containing 'train_top_ids'
        n_examples: Number of examples per class (default 2)
        seed: Random seed for reproducibility
    """
    bias_configs = [
        ("r_bias", "Red-Biased", "red"),
        ("g_bias", "Green-Biased", "green"),
        ("b_bias", "Blue-Biased", "blue"),
    ]

    n_classes = 3

    fig, axes = plt.subplots(3, n_classes * 2, figsize=(16, 10))
    np.random.seed(seed)

    for row_idx, (bias_col, bias_label, color) in enumerate(bias_configs):
        train_top_ids = color_bias_results[bias_col]["train_top_ids"][:n_classes]

        col_idx = 0
        for cid in train_top_ids:
            class_name = id_to_name.get(cid, f"ID_{cid}")[:18]

            train_samples = df[(df["split"] == "train") & (df["class_id"] == cid)]
            if len(train_samples) > 0:
                sample = train_samples.nlargest(5, bias_col).sample(1).iloc[0]
                img = ds_train["images"][int(sample["idx"])].numpy()
                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].set_title(f"Train: {class_name}", fontsize=8)
            axes[row_idx, col_idx].axis("off")
            col_idx += 1

            val_samples = df[(df["split"] == "val") & (df["class_id"] == cid)]
            if len(val_samples) > 0:
                sample = val_samples.nlargest(5, bias_col).sample(1).iloc[0]
                img = ds_val["images"][int(sample["idx"])].numpy()
                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].set_title(f"Val: {class_name}", fontsize=8)
            else:
                axes[row_idx, col_idx].text(
                    0.5, 0.5, "No Val\nsamples", ha="center", va="center"
                )
            axes[row_idx, col_idx].axis("off")
            col_idx += 1

        axes[row_idx, 0].annotate(
            bias_label,
            xy=(-0.3, 0.5),
            xycoords="axes fraction",
            fontsize=11,
            fontweight="bold",
            color=color,
            rotation=90,
            va="center",
            ha="center",
        )

    plt.suptitle(
        "Top Color-Biased Classes: Example Images (Train vs Val)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_top_classes_by_color_bias(
    df: pd.DataFrame,
    id_to_name: Dict[int, str],
    n_classes: int = 5,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Show top classes by R/G/B color bias for train and validation splits.

    Creates a 2x3 grid of horizontal barplots showing classes with the
    highest mean color bias values for each channel.

    Args:
        df: DataFrame with 'split', 'class_id', 'r_bias', 'g_bias',
            'b_bias' columns.
        id_to_name: Mapping from class_id to human-readable name.
        n_classes: Number of top classes to show per bias type.

    Returns:
        Dict[str, Dict[str, List[int]]]: Nested dictionary with structure:
            {bias_col: {'train_top_ids': [...], 'val_top_ids': [...]}}
            where bias_col is 'r_bias', 'g_bias', or 'b_bias'.
    """
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    bias_configs = [
        ("r_bias", "Red Bias", "red"),
        ("g_bias", "Green Bias", "green"),
        ("b_bias", "Blue Bias", "blue"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    results = {}

    for col_idx, (bias_col, bias_label, color) in enumerate(bias_configs):
        train_means = (
            train_df.groupby("class_id")[bias_col].mean().sort_values(ascending=False)
        )
        val_means = (
            val_df.groupby("class_id")[bias_col].mean().sort_values(ascending=False)
        )

        train_top = train_means.head(n_classes)
        val_top = val_means.head(n_classes)

        results[bias_col] = {
            "train_top_ids": list(train_top.index),
            "val_top_ids": list(val_top.index),
        }

        class_names_train = [
            id_to_name.get(cid, f"ID_{cid}")[:25] for cid in train_top.index
        ]
        axes[0, col_idx].barh(
            class_names_train[::-1], train_top.values[::-1], color=color, alpha=0.7
        )
        axes[0, col_idx].set_xlabel(bias_label)
        axes[0, col_idx].set_title(
            f"TRAIN: Top {n_classes} {bias_label}", fontsize=11, fontweight="bold"
        )
        axes[0, col_idx].grid(True, axis="x", alpha=0.3)

        class_names_val = [
            id_to_name.get(cid, f"ID_{cid}")[:25] for cid in val_top.index
        ]
        axes[1, col_idx].barh(
            class_names_val[::-1], val_top.values[::-1], color=color, alpha=0.7
        )
        axes[1, col_idx].set_xlabel(bias_label)
        axes[1, col_idx].set_title(
            f"VAL: Top {n_classes} {bias_label}", fontsize=11, fontweight="bold"
        )
        axes[1, col_idx].grid(True, axis="x", alpha=0.3)

        overlap = set(train_top.index) & set(val_top.index)
        axes[1, col_idx].set_xlabel(
            f"{bias_label}\n({len(overlap)}/{n_classes} overlap)"
        )

    plt.suptitle(
        f"Top Classes by Color Bias (Train vs Val)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    return results


def make_fn(p: A.Compose) -> Callable[[np.ndarray, DeepLakeDataset, int], np.ndarray]:
    """
    Create a wrapper function for an Albumentations pipeline.

    Converts an Albumentations Compose object into a callable that
    accepts (img, ds, idx) and returns the transformed image.

    Args:
        p: Albumentations Compose pipeline.

    Returns:
        Callable: Function with signature (img, ds, idx) -> np.ndarray.
    """
    return lambda img, ds, idx, p=p: p(image=img)["image"]
