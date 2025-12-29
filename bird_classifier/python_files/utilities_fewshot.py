from __future__ import annotations

from typing import Any, Callable, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix as sklearn_cm
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from fewshot import flatten_indices, get_labels_for_indices


class ConfusedPair(TypedDict):
    true_class: int
    pred_class: int
    count: int


class ExperimentResult(TypedDict):
    frozen_accuracy: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    history: dict[str, list[float]]
    model: "SupervisedProjectionClassifier"


class SampleInfo(TypedDict):
    ds_idx: int
    true: int
    pred: int
    correct: bool


def make_predict_fn_lime(
    model: nn.Module,
    backbone_model: nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    prototype_matrix_np_norm: npt.NDArray[np.floating[Any]],
    device: torch.device,
) -> Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.floating[Any]]]:
    """
    Factory function to create a LIME prediction function.

    Args:
        model: The projection model with get_embedding method
        backbone_model: The backbone feature extractor
        preprocess: Preprocessing transforms for the backbone
        prototype_matrix_np_norm: Normalized prototype matrix (n_classes x embedding_dim)
        device: torch device to use

    Returns:
        predict_fn: A function that takes images and returns class probabilities
    """

    def predict_fn_lime(
        images: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.floating[Any]]:
        all_probs: list[npt.NDArray[np.floating[Any]]] = []

        with torch.no_grad():
            for img in images:
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                tensor = preprocess(pil_img).unsqueeze(0).to(device)
                backbone_emb = backbone_model(tensor)
                if hasattr(model, "get_embedding"):
                    proj_emb = model.get_embedding(backbone_emb)
                else:
                    proj_emb = model(backbone_emb)
                proj_emb_np = proj_emb.cpu().numpy()
                proj_emb_norm = proj_emb_np / (
                    np.linalg.norm(proj_emb_np, axis=1, keepdims=True) + 1e-8
                )
                similarities = proj_emb_norm @ prototype_matrix_np_norm.T
                logits = similarities[0] * 10
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / exp_logits.sum()
                all_probs.append(probs)

        return np.array(all_probs)

    return predict_fn_lime


_idx_to_pos: dict[int, int] | None = None
ALL_EMBEDDINGS: npt.NDArray[np.floating[Any]] | None = None


def setup_embedding_lookup(
    embeddings: npt.NDArray[np.floating[Any]],
    indices: npt.NDArray[np.integer[Any]],
) -> None:
    """Initialize the fast embedding lookup with pre-computed embeddings.

    Sets up module-level globals for O(1) embedding retrieval by dataset index.

    Args:
        embeddings: Pre-computed embeddings array of shape (n_samples, embedding_dim).
        indices: Dataset indices corresponding to each embedding row.
    """
    global _idx_to_pos, ALL_EMBEDDINGS
    ALL_EMBEDDINGS = embeddings
    _idx_to_pos = {int(idx): pos for pos, idx in enumerate(indices)}


def get_embeddings_fast(
    indices: list[int] | npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Retrieve pre-computed embeddings by dataset indices.

    Provides O(n) lookup from the embedding cache instead of re-computing
    embeddings through the backbone model.

    Args:
        indices: Dataset indices to retrieve embeddings for.

    Returns:
        Embeddings array of shape (len(indices), embedding_dim).

    Raises:
        RuntimeError: If setup_embedding_lookup() was not called first.
    """
    if _idx_to_pos is None or ALL_EMBEDDINGS is None:
        raise RuntimeError(
            "Call setup_embedding_lookup() first to initialize embedding cache"
        )
    positions = [_idx_to_pos[int(i)] for i in indices]
    return ALL_EMBEDDINGS[positions]


def flatten_class_indices(
    indices_dict: dict[int, list[int] | npt.NDArray[np.integer[Any]]],
) -> tuple[list[int], list[int]]:
    """Flatten a dictionary of class indices into parallel lists.

    Args:
        indices_dict: Mapping from class_id to list of dataset indices.

    Returns:
        A tuple of (flat_indices, flat_labels) where flat_indices contains
        all dataset indices and flat_labels contains the corresponding class IDs.
    """
    flat_indices = []
    flat_labels = []
    for class_id, indices in indices_dict.items():
        for idx in indices:
            flat_indices.append(int(idx))
            flat_labels.append(int(class_id))
    return flat_indices, flat_labels


class ProjectionHead(nn.Module):
    """Neural network module that projects embeddings to a lower-dimensional space.

    Applies a two-layer MLP with LayerNorm, GELU activation, and dropout,
    optionally adding a scaled residual connection.

    Attributes:
        use_residual: Whether to use residual connections.
        projection: The sequential projection layers.
        residual_proj: Optional linear layer for residual dimension matching.
    """

    def __init__(
        self,
        input_dim: int = 1792,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        """Initialize the projection head.

        Args:
            input_dim: Dimension of input embeddings.
            hidden_dim: Dimension of hidden layer.
            output_dim: Dimension of output embeddings.
            dropout: Dropout probability.
            use_residual: Whether to add a scaled residual connection.
        """
        super().__init__()
        self.use_residual: bool = use_residual
        self.projection: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        if use_residual and input_dim != output_dim:
            self.residual_proj: nn.Linear | None = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embeddings to output space.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            L2-normalized projected embeddings of shape (batch_size, output_dim).
        """
        z = self.projection(x)
        if self.use_residual:
            residual = self.residual_proj(x) if self.residual_proj else x
            z = z + 0.1 * residual
        return F.normalize(z, p=2, dim=1)


class SupervisedProjectionClassifier(nn.Module):
    """Classifier combining projection head with cosine similarity classification.

    Projects embeddings through a ProjectionHead, then computes scaled cosine
    similarities against learned class weight vectors for classification.

    Attributes:
        projection: The projection head module.
        classifier: Linear layer whose weights serve as class prototypes.
        scale: Learnable temperature parameter for similarity scaling.
    """

    def __init__(
        self,
        input_dim: int = 1792,
        hidden_dim: int = 1024,
        embedding_dim: int = 512,
        n_classes: int = 555,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the classifier.

        Args:
            input_dim: Dimension of input embeddings from backbone.
            hidden_dim: Dimension of projection head hidden layer.
            embedding_dim: Dimension of projected embeddings.
            n_classes: Number of output classes.
            dropout: Dropout probability in projection head.
        """
        super().__init__()
        self.projection: ProjectionHead = ProjectionHead(
            input_dim, hidden_dim, embedding_dim, dropout, True
        )
        self.classifier: nn.Linear = nn.Linear(embedding_dim, n_classes, bias=False)
        self.scale: nn.Parameter = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute classification logits.

        Args:
            x: Input embeddings of shape (batch_size, input_dim).

        Returns:
            Logits of shape (batch_size, n_classes).
        """
        z = self.projection(x)
        w = F.normalize(self.classifier.weight, p=2, dim=1)
        return self.scale * torch.mm(z, w.t())

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract projected embeddings without classification.

        Args:
            x: Input embeddings of shape (batch_size, input_dim).

        Returns:
            Projected embeddings of shape (batch_size, embedding_dim).
        """
        return self.projection(x)


def evaluate_with_prototypes(
    model: SupervisedProjectionClassifier,
    train_embeddings: npt.NDArray[np.floating[Any]],
    train_labels: npt.NDArray[np.integer[Any]] | list[int],
    val_embeddings: npt.NDArray[np.floating[Any]],
    val_labels: npt.NDArray[np.integer[Any]] | list[int],
    device: torch.device,
    return_predictions: bool = False,
    debug: bool = False,
) -> float | tuple[float, npt.NDArray[np.int64]]:
    """Evaluate model using prototype-based classification.

    Computes class prototypes from training embeddings, then classifies
    validation samples by nearest prototype using cosine similarity.

    Args:
        model: Trained projection classifier model.
        train_embeddings: Training set embeddings of shape (n_train, embed_dim).
        train_labels: Class labels for training samples.
        val_embeddings: Validation set embeddings of shape (n_val, embed_dim).
        val_labels: Ground truth labels for validation samples.
        device: Torch device for computation.
        return_predictions: If True, return predictions along with accuracy.
        debug: If True, print debug information.

    Returns:
        If return_predictions is False, returns accuracy as a float.
        If return_predictions is True, returns tuple of (accuracy, predictions).
    """
    model.eval()
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    val_labels = np.asarray(val_labels, dtype=np.int64).reshape(-1)

    unique_classes = np.unique(train_labels)
    n_classes = len(unique_classes)
    class_to_idx = {int(c): i for i, c in enumerate(unique_classes)}

    if debug:
        print(
            f"DEBUG: n_classes={n_classes}, train_labels sample: {train_labels[:5]}, val_labels sample: {val_labels[:5]}"
        )

    with torch.no_grad():
        train_emb_t = torch.from_numpy(train_embeddings).float().to(device)
        train_proj = model.get_embedding(train_emb_t)

        prototypes = torch.zeros(n_classes, train_proj.shape[1], device=device)
        counts = torch.zeros(n_classes, device=device)

        for i, label in enumerate(train_labels):
            idx = class_to_idx[int(label)]
            prototypes[idx] += train_proj[i]
            counts[idx] += 1

        prototypes = prototypes / counts.unsqueeze(1).clamp(min=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)

        val_emb_t = torch.from_numpy(val_embeddings).float()
        all_preds = []

        for i in range(0, len(val_emb_t), 256):
            batch = val_emb_t[i : i + 256].to(device)
            batch_proj = model.get_embedding(batch)
            similarities = torch.mm(batch_proj, prototypes.t())
            preds = similarities.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    idx_to_class = {i: c for c, i in class_to_idx.items()}
    pred_labels = np.array([idx_to_class[int(p)] for p in all_preds], dtype=np.int64)
    accuracy = (pred_labels == val_labels).mean()

    if debug:
        print(f"DEBUG: pred_labels sample: {pred_labels[:5]}, accuracy: {accuracy:.4f}")

    model.train()
    return (accuracy, pred_labels) if return_predictions else accuracy


def evaluate_frozen_baseline(
    train_embeddings: npt.NDArray[np.floating[Any]],
    train_labels: npt.NDArray[np.integer[Any]] | list[int],
    val_embeddings: npt.NDArray[np.floating[Any]],
    val_labels: npt.NDArray[np.integer[Any]] | list[int],
) -> float:
    """Evaluate frozen backbone embeddings using prototype classification.

    Computes class prototypes directly from backbone embeddings without
    any learned projection, serving as a baseline comparison.

    Args:
        train_embeddings: Training embeddings of shape (n_train, embed_dim).
        train_labels: Class labels for training samples.
        val_embeddings: Validation embeddings of shape (n_val, embed_dim).
        val_labels: Ground truth labels for validation samples.

    Returns:
        Classification accuracy on validation set.
    """
    train_labels = np.array(train_labels, dtype=np.int64)
    val_labels = np.array(val_labels, dtype=np.int64)

    unique_classes = np.unique(train_labels)
    class_to_idx = {int(c): i for i, c in enumerate(unique_classes)}

    prototypes = np.zeros((len(unique_classes), train_embeddings.shape[1]))
    counts = np.zeros(len(unique_classes))

    for i, label in enumerate(train_labels):
        idx = class_to_idx[int(label)]
        prototypes[idx] += train_embeddings[i]
        counts[idx] += 1

    prototypes = prototypes / counts[:, np.newaxis].clip(min=1)
    prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    val_norm = val_embeddings / (
        np.linalg.norm(val_embeddings, axis=1, keepdims=True) + 1e-8
    )

    similarities = val_norm @ prototypes.T
    pred_indices = similarities.argmax(axis=1)

    idx_to_class = {i: c for c, i in class_to_idx.items()}
    pred_labels = np.array([idx_to_class[int(p)] for p in pred_indices], dtype=np.int64)

    return (pred_labels == val_labels).mean()


def train_supervised_projection(
    model: SupervisedProjectionClassifier,
    train_embeddings: npt.NDArray[np.floating[Any]],
    train_labels: npt.NDArray[np.integer[Any]],
    train_labels_original: npt.NDArray[np.integer[Any]] | list[int],
    val_embeddings: npt.NDArray[np.floating[Any]] | None,
    val_labels: npt.NDArray[np.integer[Any]] | list[int] | None,
    device: torch.device,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train projection classifier with cross-entropy loss.

    Uses AdamW optimizer with cosine annealing learning rate schedule.
    Validates every 5 epochs using prototype-based evaluation and restores
    the best model checkpoint at the end.

    Args:
        model: Projection classifier model to train.
        train_embeddings: Training embeddings of shape (n_train, input_dim).
        train_labels: Mapped labels (0 to N-1) for cross-entropy loss.
        train_labels_original: Original class labels for prototype evaluation.
        val_embeddings: Validation embeddings, or None to skip validation.
        val_labels: Validation labels, or None to skip validation.
        device: Torch device for training.
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        label_smoothing: Label smoothing factor for cross-entropy loss.
        verbose: Whether to show progress bar.

    Returns:
        Training history dict with keys 'epoch', 'train_loss', 'train_acc', 'val_acc'.
    """
    model = model.to(device)
    train_labels_original = np.asarray(train_labels_original, dtype=np.int64).reshape(
        -1
    )
    if val_labels is not None:
        val_labels = np.asarray(val_labels, dtype=np.int64).reshape(-1)

    train_dataset = TensorDataset(
        torch.from_numpy(train_embeddings).float(),
        torch.from_numpy(train_labels).long(),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    pbar = tqdm(range(n_epochs), desc="Training", disable=not verbose)

    for epoch in pbar:
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for batch_emb, batch_labels in train_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_emb)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_labels)
            epoch_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            epoch_total += len(batch_labels)

        scheduler.step()

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_embeddings is not None and (epoch + 1) % 5 == 0:
            val_acc = evaluate_with_prototypes(
                model,
                train_embeddings,
                train_labels_original,
                val_embeddings,
                val_labels,
                device,
            )
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            pbar.set_postfix(
                {
                    "loss": f"{train_loss:.4f}",
                    "train": f"{train_acc*100:.1f}%",
                    "val": f"{val_acc*100:.1f}%",
                }
            )
        else:
            pbar.set_postfix(
                {"loss": f"{train_loss:.4f}", "train": f"{train_acc*100:.1f}%"}
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model with val accuracy = {best_val_acc*100:.2f}%")

    return history


def run_learned_projection_experiment(
    extractor: Any,
    ds_train: Any,
    support_indices: dict[int, npt.NDArray[np.int64] | list[int]],
    pool_indices: dict[int, npt.NDArray[np.int64] | list[int]],
    val_indices: dict[int, npt.NDArray[np.int64] | list[int]],
    device: torch.device,
    batch_size: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
    embedding_dim: int = 512,
    use_pool_data: bool = False,
    seed: int = 42,
    embedding_lookup_fn: (
        Callable[[list[int]], npt.NDArray[np.floating[Any]]] | None
    ) = None,
) -> ExperimentResult:
    """Run complete learned projection experiment with training and evaluation.

    Trains a projection head on support (and optionally pool) data, evaluates
    against frozen baseline, and reports comprehensive metrics with visualizations.

    Args:
        extractor: Feature extractor with extract_from_dataset method.
        ds_train: Training dataset with 'images' and 'labels' keys.
        support_indices: Dict mapping class_id to support set indices.
        pool_indices: Dict mapping class_id to pool set indices.
        val_indices: Dict mapping class_id to validation set indices.
        device: Torch device for training and inference.
        batch_size: Batch size for embedding extraction and training.
        n_epochs: Number of training epochs.
        lr: Learning rate for optimizer.
        embedding_dim: Output dimension of projection head.
        use_pool_data: Whether to include pool data in training.
        seed: Random seed for reproducibility.
        embedding_lookup_fn: Optional function for fast embedding lookup.

    Returns:
        ExperimentResult containing accuracy, metrics, history, and trained model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("LEARNED PROJECTION EXPERIMENT")
    print("=" * 70)

    # Prepare data
    if use_pool_data:
        train_indices_dict = {}
        for cid in set(support_indices.keys()) | set(pool_indices.keys()):
            support = list(support_indices.get(cid, []))
            pool = list(pool_indices.get(cid, []))
            train_indices_dict[cid] = np.array(support + pool, dtype=np.int64)
    else:
        train_indices_dict = support_indices

    train_flat = flatten_indices(train_indices_dict)
    val_flat = flatten_indices(val_indices)

    print(f"\n[1/5] Data prepared")
    print(f"  Training samples: {len(train_flat)}")
    print(f"  Validation samples: {len(val_flat)}")

    # Extract embeddings - use cache if available
    print("\n[2/5] Extracting embeddings...")
    if embedding_lookup_fn is not None:
        print("  Using pre-computed embeddings (instant lookup)")
        train_embeddings = embedding_lookup_fn(train_flat)
        val_embeddings = embedding_lookup_fn(val_flat)
    else:
        train_embeddings = extractor.extract_from_dataset(
            ds_train, train_flat, batch_size
        )
        val_embeddings = extractor.extract_from_dataset(ds_train, val_flat, batch_size)

    train_labels = get_labels_for_indices(ds_train, train_flat)
    val_labels = get_labels_for_indices(ds_train, val_flat)

    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Val embeddings: {val_embeddings.shape}")

    # Map labels for cross-entropy
    unique_classes = np.unique(train_labels)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    train_labels_mapped = np.array([class_to_idx[l] for l in train_labels])

    # Frozen baseline
    print("\n[3/5] Frozen baseline...")
    frozen_acc = evaluate_frozen_baseline(
        train_embeddings, train_labels, val_embeddings, val_labels
    )
    print(f"  Frozen accuracy: {frozen_acc*100:.2f}%")

    # Train
    print("\n[4/5] Training projection...")
    model = SupervisedProjectionClassifier(
        input_dim=train_embeddings.shape[1],
        hidden_dim=1024,
        embedding_dim=embedding_dim,
        n_classes=len(unique_classes),
        dropout=0.1,
    )

    history = train_supervised_projection(
        model=model,
        train_embeddings=train_embeddings,
        train_labels=train_labels_mapped,
        train_labels_original=train_labels,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        device=device,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
    )

    # Final eval
    print("\n[5/5] Final evaluation...")
    final_acc, pred_labels = evaluate_with_prototypes(
        model,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        device,
        return_predictions=True,
    )

    val_precision = precision_score(
        val_labels, pred_labels, average="macro", zero_division=0
    )
    val_recall = recall_score(val_labels, pred_labels, average="macro", zero_division=0)
    val_f1 = f1_score(val_labels, pred_labels, average="macro", zero_division=0)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Validation Accuracy: {final_acc*100:.2f}%")
    print(
        f"  Precision: {val_precision*100:.2f}%  Recall: {val_recall*100:.2f}%  F1: {val_f1*100:.2f}%"
    )
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        history["epoch"],
        [a * 100 for a in history["train_acc"]],
        "b-",
        alpha=0.7,
        label="Train",
    )
    if history["val_acc"]:
        val_epochs = list(range(5, n_epochs + 1, 5))[: len(history["val_acc"])]
        axes[1].plot(
            val_epochs, [a * 100 for a in history["val_acc"]], "ro-", label="Validation"
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "frozen_accuracy": frozen_acc,
        "val_accuracy": final_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "history": history,
        "model": model,
    }


def compute_confused_pairs(
    y_true: npt.NDArray[np.integer[Any]] | list[int],
    y_pred: npt.NDArray[np.integer[Any]] | list[int],
    top_n: int = 20,
) -> list[ConfusedPair]:
    """Identify the most frequently confused class pairs from predictions.

    Builds a confusion matrix and extracts off-diagonal entries representing
    misclassifications, sorted by frequency.

    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        top_n: Maximum number of confused pairs to return.

    Returns:
        List of ConfusedPair dicts sorted by count descending, each containing
        'true_class', 'pred_class', and 'count' keys.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = sklearn_cm(y_true, y_pred, labels=unique_classes)

    confused_pairs: list[ConfusedPair] = []
    for i, true_cls in enumerate(unique_classes):
        for j, pred_cls in enumerate(unique_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append(
                    {
                        "true_class": int(true_cls),
                        "pred_class": int(pred_cls),
                        "count": int(cm[i, j]),
                    }
                )

    confused_pairs.sort(key=lambda x: -x["count"])
    return confused_pairs[:top_n]


def show_class_samples(
    class_items: list[tuple[int, dict[str, Any]]],
    title: str,
    ds_train: Any | None = None,
    ds_val: Any | None = None,
    class_names: list[str] | None = None,
    holdout_sample_info: list[SampleInfo] | None = None,
    show_mistakes: bool = False,
    n_samples: int = 3,
) -> None:
    """Display a grid of sample images for specified classes.

    Creates a matplotlib figure with one row per class, showing a reference
    image from training data and additional samples from holdout data.

    Args:
        class_items: List of (class_id, stats_dict) tuples where stats_dict
            may contain 'accuracy', 'total', 'correct', and 'support_size' keys.
        title: Title for the figure.
        ds_train: Training dataset for reference images. Must support indexing
            with 'images' and 'labels' keys.
        ds_val: Validation dataset for sample images.
        class_names: List of class names indexed by class_id.
        holdout_sample_info: List of SampleInfo dicts with sample metadata.
        show_mistakes: If True, show misclassified examples; otherwise show
            correctly classified examples.
        n_samples: Number of sample images to show per class.
    """
    if class_names is None:
        class_names = []

    n_classes = len(class_items)
    if n_classes == 0:
        print("No classes to display")
        return

    n_cols = n_samples + 1
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(4 * n_cols, 4 * n_classes))

    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for row, (class_id, stats) in enumerate(class_items):
        class_id = int(class_id)
        acc = stats.get("accuracy", 0)
        total = stats.get("total", stats.get("support_size", 0))
        correct = stats.get("correct", int(acc * total))

        # Get class name
        if class_id < len(class_names):
            name = class_names[class_id]
            short_name = name.split("/")[-1] if "/" in name else name
        else:
            short_name = f"Class_{class_id}"

        # Reference image from training set
        ax = axes[row, 0]
        if ds_train is not None:
            try:
                # Find an example of this class in training data
                for idx in range(min(1000, len(ds_train))):
                    if int(ds_train["labels"][idx].numpy()) == class_id:
                        img = ds_train["images"][idx].numpy()
                        ax.imshow(img)
                        break
            except:
                pass
        ax.set_title(
            f"Ref: {short_name[:25]}\nAcc: {acc*100:.1f}% ({correct}/{total})",
            fontsize=9,
        )
        ax.axis("off")

        if show_mistakes and holdout_sample_info is not None:
            # Show misclassified examples
            mistakes = [
                s
                for s in holdout_sample_info
                if s["true"] == class_id and not s["correct"]
            ]
            for col in range(1, n_cols):
                ax = axes[row, col]
                if col - 1 < len(mistakes) and ds_val is not None:
                    try:
                        ds_idx = mistakes[col - 1]["ds_idx"]
                        pred = mistakes[col - 1]["pred"]
                        img = ds_val["images"][ds_idx].numpy()
                        ax.imshow(img)
                        pred_name = (
                            class_names[pred].split("/")[-1]
                            if pred < len(class_names)
                            else f"Class_{pred}"
                        )
                        ax.set_title(f"Pred: {pred_name[:20]}", fontsize=8, color="red")
                    except:
                        pass
                ax.axis("off")
        else:
            # Show correct examples from holdout
            if holdout_sample_info is not None:
                correct_samples = [
                    s
                    for s in holdout_sample_info
                    if s["true"] == class_id and s["correct"]
                ]
            else:
                correct_samples = []
            for col in range(1, n_cols):
                ax = axes[row, col]
                if col - 1 < len(correct_samples) and ds_val is not None:
                    try:
                        ds_idx = correct_samples[col - 1]["ds_idx"]
                        img = ds_val["images"][ds_idx].numpy()
                        ax.imshow(img)
                        ax.set_title("Correct", fontsize=8, color="green")
                    except:
                        pass
                ax.axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
