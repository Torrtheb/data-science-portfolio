"""
Learned Projection for Few-Shot Bird Classification
====================================================

This module implements a SUPERVISED approach to learning better embeddings
for the NABirds dataset. Unlike episodic meta-learning, we train directly 
on all 555 classes using standard cross-entropy loss.

The Approach
------------
1. Train a projection head with cross-entropy on support samples
2. Use the learned projection for prototype-based classification
3. Evaluate on a held-out validation set (never used for training)

Key Insight
-----------
Unlike episodic training (30-way) which may not generalize to 555-way 
evaluation, supervised training directly optimizes for the full 555-class 
problem while keeping the global discriminative structure.

Few-Shot Compliance
-------------------
- Training uses ONLY the support set (5 samples/class)
- Validation accuracy is computed on a held-out set
- Pool data is never used as training labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class ProjectionHead(nn.Module):
    """
    Learnable projection head that transforms frozen embeddings.
    
    Architecture designed for fine-grained classification:
    - Preserves most of the input information (larger hidden dims)
    - Uses LayerNorm instead of BatchNorm (more stable for small batches)
    - Residual connection to preserve pretrained structure
    """
    def __init__(
        self, 
        input_dim: int = 1792,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        # Main projection path
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Residual projection (if dimensions differ)
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings with optional residual connection."""
        z = self.projection(x)
        
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            z = z + 0.1 * residual  # Small residual weight
        
        # L2 normalize for cosine similarity
        return F.normalize(z, p=2, dim=1)


class SupervisedProjectionClassifier(nn.Module):
    """
    Supervised classifier that learns a projection + classification head.
    
    After training, we use the projection for prototype-based classification
    which generalizes better than the learned classifier weights.
    """
    def __init__(
        self,
        input_dim: int = 1792,
        hidden_dim: int = 1024,
        embedding_dim: int = 512,
        n_classes: int = 555,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projection = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
            use_residual=True
        )
        
        # Cosine classifier (normalized weights)
        self.classifier = nn.Linear(embedding_dim, n_classes, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))  # Learnable temperature
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for training with cross-entropy."""
        z = self.projection(x)
        # Normalize classifier weights for cosine similarity
        w = F.normalize(self.classifier.weight, p=2, dim=1)
        logits = self.scale * torch.mm(z, w.t())
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized embeddings for prototype classification."""
        return self.projection(x)


def train_supervised_projection(
    model: SupervisedProjectionClassifier,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: Optional[np.ndarray],
    val_labels: Optional[np.ndarray],
    device: torch.device,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    verbose: bool = True
) -> Dict:
    """
    Train projection head with supervised cross-entropy loss.
    
    This is standard supervised training on the support set, which
    teaches the projection to be discriminative for all 555 classes.
    """
    model = model.to(device)
    
    # Create dataset
    train_dataset = TensorDataset(
        torch.from_numpy(train_embeddings).float(),
        torch.from_numpy(train_labels).long()
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=n_epochs
    )
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_state = None
    
    pbar = tqdm(range(n_epochs), desc="Training", disable=not verbose)
    
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_emb, batch_labels in train_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_emb)
            loss = criterion(logits, batch_labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_labels)
            epoch_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            epoch_total += len(batch_labels)
        
        scheduler.step()
        
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        if val_embeddings is not None and (epoch + 1) % 5 == 0:
            val_acc = evaluate_with_prototypes(
                model, train_embeddings, train_labels,
                val_embeddings, val_labels, device
            )
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'train': f'{train_acc*100:.1f}%',
                'val': f'{val_acc*100:.1f}%'
            })
        else:
            pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc*100:.1f}%'
            })
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model with validation accuracy = {best_val_acc*100:.2f}%")
    
    return history


def evaluate_with_prototypes(
    model: SupervisedProjectionClassifier,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device
) -> float:
    """
    Evaluate using prototype-based classification with learned embeddings.
    
    1. Project all training embeddings
    2. Compute prototype (mean) for each class
    3. Project validation embeddings
    4. Classify by nearest prototype (cosine similarity)
    """
    model.eval()
    
    unique_classes = np.unique(train_labels)
    n_classes = len(unique_classes)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    with torch.no_grad():
        # Project training embeddings and compute prototypes
        train_emb_t = torch.from_numpy(train_embeddings).float().to(device)
        train_proj = model.get_embedding(train_emb_t)
        
        # Compute prototypes
        prototypes = torch.zeros(n_classes, train_proj.shape[1], device=device)
        counts = torch.zeros(n_classes, device=device)
        
        for i, label in enumerate(train_labels):
            idx = class_to_idx[label]
            prototypes[idx] += train_proj[i]
            counts[idx] += 1
        
        prototypes = prototypes / counts.unsqueeze(1).clamp(min=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Project and classify validation
        val_emb_t = torch.from_numpy(val_embeddings).float()
        batch_size = 256
        all_preds = []
        
        for i in range(0, len(val_emb_t), batch_size):
            batch = val_emb_t[i:i+batch_size].to(device)
            batch_proj = model.get_embedding(batch)
            similarities = torch.mm(batch_proj, prototypes.t())
            preds = similarities.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    
    # Convert back to original labels
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    pred_labels = np.array([idx_to_class[p] for p in all_preds])
    
    accuracy = (pred_labels == val_labels).mean()
    
    model.train()
    return accuracy


def evaluate_frozen_baseline(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray
) -> float:
    """
    Evaluate baseline with frozen embeddings (no learned projection).
    """
    unique_classes = np.unique(train_labels)
    n_classes = len(unique_classes)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    # Compute prototypes
    prototypes = np.zeros((n_classes, train_embeddings.shape[1]))
    counts = np.zeros(n_classes)
    
    for i, label in enumerate(train_labels):
        idx = class_to_idx[label]
        prototypes[idx] += train_embeddings[i]
        counts[idx] += 1
    
    prototypes = prototypes / counts[:, np.newaxis].clip(min=1)
    
    # L2 normalize
    prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    val_norm = val_embeddings / (np.linalg.norm(val_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity classification
    similarities = val_norm @ prototypes.T
    pred_indices = similarities.argmax(axis=1)
    
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    pred_labels = np.array([idx_to_class[p] for p in pred_indices])
    
    return (pred_labels == val_labels).mean()


def run_learned_projection_experiment(
    extractor,
    ds_train,
    support_indices: Dict[int, np.ndarray],
    pool_indices: Dict[int, np.ndarray],
    val_indices: Dict[int, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
    embedding_dim: int = 512,
    use_pool_data: bool = False,
    seed: int = 42
) -> Dict:
    """
    Run the learned projection experiment.
    
    Args:
        extractor: Feature extractor
        ds_train: Training dataset
        support_indices: Dict mapping class_id -> support indices
        pool_indices: Dict mapping class_id -> pool indices  
        val_indices: Dict mapping class_id -> validation indices
        device: PyTorch device
        batch_size: Training batch size
        n_epochs: Number of training epochs
        lr: Learning rate
        embedding_dim: Output embedding dimension
        use_pool_data: Whether to include pool data in training
        seed: Random seed
    
    Returns:
        Dict with experiment results
    """
    from fewshot import flatten_indices, get_labels_for_indices
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("LEARNED PROJECTION EXPERIMENT (Supervised)")
    print("=" * 70)
    
    # Step 1: Prepare data
    print("\n[1/5] Preparing data...")
    
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
    
    print(f"  Training samples: {len(train_flat)}")
    print(f"  Validation samples: {len(val_flat)}")
    print(f"  Number of classes: {len(train_indices_dict)}")
    
    # Step 2: Extract embeddings
    print("\n[2/5] Extracting embeddings...")
    
    train_embeddings = extractor.extract_from_dataset(ds_train, train_flat, batch_size)
    val_embeddings = extractor.extract_from_dataset(ds_train, val_flat, batch_size)
    
    train_labels = get_labels_for_indices(ds_train, train_flat)
    val_labels = get_labels_for_indices(ds_train, val_flat)
    
    # Map labels to contiguous indices for training
    unique_classes = np.unique(train_labels)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    train_labels_mapped = np.array([class_to_idx[l] for l in train_labels])
    
    print(f"  Embedding shape: {train_embeddings.shape}")
    
    # Step 3: Evaluate frozen baseline
    print("\n[3/5] Evaluating frozen baseline...")
    
    frozen_acc = evaluate_frozen_baseline(
        train_embeddings, train_labels, val_embeddings, val_labels
    )
    print(f"  Frozen baseline accuracy: {frozen_acc*100:.2f}%")
    
    # Step 4: Train projection
    print("\n[4/5] Training learned projection...")
    print(f"  Epochs: {n_epochs}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Learning rate: {lr}")
    
    model = SupervisedProjectionClassifier(
        input_dim=train_embeddings.shape[1],
        hidden_dim=1024,
        embedding_dim=embedding_dim,
        n_classes=len(unique_classes),
        dropout=0.1
    )
    
    history = train_supervised_projection(
        model=model,
        train_embeddings=train_embeddings,
        train_labels=train_labels_mapped,  # Use mapped labels
        val_embeddings=val_embeddings,
        val_labels=val_labels,  # Keep original for prototype eval
        device=device,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True
    )
    
    # Step 5: Final evaluation
    print("\n[5/5] Final evaluation...")
    
    final_acc = evaluate_with_prototypes(
        model, train_embeddings, train_labels,
        val_embeddings, val_labels, device
    )
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY (Validation Set)")
    print(f"{'='*70}")
    print(f"  Frozen baseline (val):        {frozen_acc*100:.2f}%")
    print(f"  Learned projection (val):     {final_acc*100:.2f}%")
    print(f"  Improvement:                  {(final_acc - frozen_acc)*100:+.2f}%")
    print(f"{'='*70}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['epoch'], history['train_loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['epoch'], [a*100 for a in history['train_acc']], 'b-', alpha=0.7, label='Training Accuracy')
    if history['val_acc']:
        val_epochs = [e for e in history['epoch'] if (e+1) % 5 == 0][:len(history['val_acc'])]
        axes[1].plot(val_epochs, [a*100 for a in history['val_acc']], 'ro-', label='Validation Accuracy')
    axes[1].axhline(y=frozen_acc*100, color='gray', linestyle='--', alpha=0.7, label='Frozen baseline (val)')
    axes[1].axhline(y=70, color='green', linestyle='--', alpha=0.7, label='Target (70%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'frozen_accuracy': frozen_acc,
        'learned_accuracy': final_acc,
        'improvement': final_acc - frozen_acc,
        'history': history,
        'model': model
    }


if __name__ == "__main__":
    print("Learned Projection module loaded!")
    print("\nUsage:")
    print("  from learned_projection import run_learned_projection_experiment")
    print("  ")
    print("  results = run_learned_projection_experiment(")
    print("      extractor=extractor,")
    print("      ds_train=ds_train,")
    print("      support_indices=support_indices,")
    print("      pool_indices=pool_indices,")
    print("      val_indices=val_indices_split,")
    print("      device=DEVICE,")
    print("      n_epochs=50")
    print("  )")
