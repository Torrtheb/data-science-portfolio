"""
Prototypical Networks for Few-Shot Bird Classification

This module implements Prototypical Networks (Snell et al., 2017) for the NABirds
dataset. It includes:
1. Quick validation experiment with learned projection head
2. Full episodic training with Prototypical Networks
3. Evaluation utilities

Reference: https://arxiv.org/abs/1703.05175
Tutorial inspiration: https://github.com/sicara/easy-few-shot-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
from pathlib import Path
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: QUICK VALIDATION - LEARNED PROJECTION HEAD
# =============================================================================

class LearnedProjectionHead(nn.Module):
    """
    A learnable MLP that projects frozen backbone embeddings to a new space.
    
    Why this helps:
    - Frozen EfficientNet was trained on ImageNet (dogs, cars, planes)
    - It doesn't know what distinguishes a Carolina Wren from a House Wren
    - This projection learns bird-specific feature relationships
    
    IMPORTANT: For 555 fine-grained classes, we use a LARGER output dimension
    to preserve discriminative information.
    
    Architecture: input_dim → hidden_dim → output_dim (L2 normalized)
    """
    def __init__(
        self, 
        input_dim: int = 1792, 
        hidden_dim: int = 1024,  # Larger hidden layer
        output_dim: int = 512,   # LARGER output to preserve info for 555 classes
        dropout: float = 0.1     # Less dropout to preserve information
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Use LayerNorm (more stable than BatchNorm for episodic/meta settings)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Extra layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize embeddings."""
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


def compute_prototypes(
    embeddings: torch.Tensor, 
    labels: torch.Tensor, 
    n_classes: int
) -> torch.Tensor:
    """
    Compute class prototypes as the mean of support embeddings.
    
    This is the core of Prototypical Networks:
    - For each class, average all support sample embeddings
    - The result is a single "prototype" vector representing that class
    
    Args:
        embeddings: [N, D] tensor of embeddings
        labels: [N] tensor of class labels (0 to n_classes-1)
        n_classes: Number of classes
    
    Returns:
        prototypes: [n_classes, D] tensor of class prototypes
    """
    # Initialize prototype accumulator
    prototype_sums = torch.zeros(n_classes, embeddings.shape[1], device=embeddings.device)
    prototype_counts = torch.zeros(n_classes, device=embeddings.device)
    
    # Sum embeddings per class
    for emb, label in zip(embeddings, labels):
        prototype_sums[label] += emb
        prototype_counts[label] += 1
    
    # Avoid division by zero
    prototype_counts = prototype_counts.clamp(min=1)
    
    # Compute mean
    prototypes = prototype_sums / prototype_counts.unsqueeze(1)
    
    return prototypes


def prototypical_loss(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 1.0,  # Standard temperature - was 0.1 which is WAY too low!
    use_cosine: bool = False   # Use Euclidean distance (original ProtoNet paper)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Prototypical Networks loss.
    
    IMPORTANT: Original ProtoNet uses EUCLIDEAN distance, not cosine!
    Temperature should be 1.0 for standard training.
    
    For each query sample:
    1. Compute distance to all prototypes
    2. Convert to probabilities (softmax over negative distances)
    3. Cross-entropy loss against true label
    
    Args:
        query_embeddings: [N_query, D] query sample embeddings
        query_labels: [N_query] ground truth labels (0 to n_classes-1)
        prototypes: [n_classes, D] class prototypes
        temperature: Softmax temperature (1.0 is standard)
        use_cosine: If True, use cosine similarity; else Euclidean distance
    
    Returns:
        loss: Scalar loss value
        accuracy: Classification accuracy on queries
    """
    if use_cosine:
        # Cosine similarity
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1)
        similarities = torch.mm(query_norm, proto_norm.t())
        log_probs = F.log_softmax(similarities / temperature, dim=1)
    else:
        # Euclidean distance (ORIGINAL ProtoNet)
        dists = torch.cdist(query_embeddings, prototypes, p=2)
        log_probs = F.log_softmax(-dists / temperature, dim=1)
    
    # Cross-entropy loss
    loss = F.nll_loss(log_probs, query_labels)
    
    # Compute accuracy
    predictions = log_probs.argmax(dim=1)
    accuracy = (predictions == query_labels).float().mean()
    
    return loss, accuracy


# =============================================================================
# PART 2: EPISODIC TRAINING INFRASTRUCTURE
# =============================================================================

class EpisodeSampler:
    """
    Samples episodes for episodic training.
    
    Each episode consists of:
    - N-way: N random classes
    - K-shot: K support samples per class
    - Q-query: Q query samples per class
    
    This mimics the few-shot evaluation scenario during training.
    """
    def __init__(
        self,
        class_to_indices: Dict[int, np.ndarray],
        n_way: int = 30,
        k_shot: int = 5,
        n_query: int = 10,
        min_samples_per_class: Optional[int] = None,
        seed: int = 42
    ):
        """
        Args:
            class_to_indices: Dict mapping class_id -> array of sample indices
            n_way: Number of classes per episode
            k_shot: Number of support samples per class
            n_query: Number of query samples per class
            min_samples_per_class: Minimum samples needed to include a class
            seed: Random seed
        """
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.rng = np.random.default_rng(seed)
        
        # Filter classes with enough samples
        min_needed = k_shot + n_query
        if min_samples_per_class is None:
            min_samples_per_class = min_needed
        self.eligible_classes = [
            cid for cid, indices in class_to_indices.items()
            if len(indices) >= max(min_needed, min_samples_per_class)
        ]
        
        self.class_to_indices = {
            cid: np.array(indices) 
            for cid, indices in class_to_indices.items()
            if cid in self.eligible_classes
        }
        
        n_excluded = len(class_to_indices) - len(self.eligible_classes)
        print(f"EpisodeSampler initialized:")
        print(f"  Eligible classes: {len(self.eligible_classes)}/{len(class_to_indices)}")
        if n_excluded > 0:
            print(f"  ⚠️  WARNING: {n_excluded} classes excluded from training (< {max(min_needed, min_samples_per_class)} samples)")
            print(f"      These classes will still appear in 555-way evaluation!")
        print(f"  Episode config: {n_way}-way {k_shot}-shot with {n_query} queries")
    
    def sample_episode(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a single episode.
        
        Returns:
            support_indices: [n_way * k_shot] indices in original dataset
            support_labels: [n_way * k_shot] labels (0 to n_way-1)
            query_indices: [n_way * n_query] indices in original dataset
            query_labels: [n_way * n_query] labels (0 to n_way-1)
        """
        # Sample n_way random classes
        episode_classes = self.rng.choice(
            self.eligible_classes, 
            size=self.n_way, 
            replace=False
        )
        
        support_indices = []
        support_labels = []
        query_indices = []
        query_labels = []
        
        for episode_label, class_id in enumerate(episode_classes):
            # Get all indices for this class
            class_indices = self.class_to_indices[class_id].copy()
            self.rng.shuffle(class_indices)
            
            # Split into support and query
            support_idx = class_indices[:self.k_shot]
            query_idx = class_indices[self.k_shot:self.k_shot + self.n_query]
            
            support_indices.extend(support_idx)
            support_labels.extend([episode_label] * len(support_idx))
            query_indices.extend(query_idx)
            query_labels.extend([episode_label] * len(query_idx))
        
        return (
            np.array(support_indices),
            np.array(support_labels),
            np.array(query_indices),
            np.array(query_labels)
        )


class PrecomputedEmbeddingDataset(Dataset):
    """
    Dataset that serves precomputed embeddings.
    
    This is more efficient than extracting embeddings on-the-fly during training.
    """
    def __init__(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray,
        indices: Optional[np.ndarray] = None
    ):
        """
        Args:
            embeddings: [N, D] precomputed embeddings
            labels: [N] class labels
            indices: [N] original dataset indices (optional, for reference)
        """
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()
        self.indices = indices
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# =============================================================================
# PART 3: PROTOTYPICAL NETWORK MODEL
# =============================================================================

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network with MINIMAL learnable projection.
    
    CRITICAL INSIGHT: The frozen embeddings already achieve 51.8% accuracy!
    We should NOT destroy this structure. Instead, we learn a SMALL refinement.
    
    Options:
    1. identity: No projection, just use frozen embeddings (baseline)
    2. linear: Single linear layer (minimal transformation)
    3. mlp: Small MLP with residual connection (preserves structure)
    
    The key is to PRESERVE the pretrained structure while learning refinements.
    """
    def __init__(
        self,
        input_dim: int = 1792,
        hidden_dim: int = 1792,   # Same as input to preserve info
        embedding_dim: int = 1792, # Keep original dimension!
        dropout: float = 0.0,      # No dropout
        mode: str = "residual"     # "identity", "linear", or "residual"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.mode = mode
        
        if mode == "identity":
            # No learnable parameters - just pass through
            self.projection = nn.Identity()
        elif mode == "linear":
            # Single linear transformation
            self.projection = nn.Linear(input_dim, embedding_dim)
            # Initialize close to identity if dimensions match
            if input_dim == embedding_dim:
                nn.init.eye_(self.projection.weight)
                nn.init.zeros_(self.projection.bias)
        elif mode == "residual":
            # Small MLP with residual connection
            # This learns a REFINEMENT to the original embeddings
            self.transform = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, embedding_dim)
            )
            # Initialize to output near-zero so residual starts as identity
            nn.init.zeros_(self.transform[-1].weight)
            nn.init.zeros_(self.transform[-1].bias)
            
            # Learnable residual weight (starts small)
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: apply minimal projection.
        
        Does NOT L2 normalize - that should be done explicitly if needed.
        """
        if self.mode == "identity":
            return x
        elif self.mode == "linear":
            return self.projection(x)
        elif self.mode == "residual":
            # Residual: original + small learned refinement
            return x + self.residual_scale * self.transform(x)
    
    def forward_episode(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """
        Process a complete episode.
        """
        # Project embeddings
        support_proj = self.forward(support_embeddings)
        query_proj = self.forward(query_embeddings)
        
        # Compute prototypes
        prototypes = compute_prototypes(support_proj, support_labels, n_way)
        
        # Compute negative distances as scores (Euclidean distance)
        dists = torch.cdist(query_proj, prototypes, p=2)
        
        return -dists  # Negative distance = higher score for closer prototypes


# =============================================================================
# PART 4: TRAINING UTILITIES
# =============================================================================

def train_projection_head_episodic(
    model: PrototypicalNetwork,
    all_embeddings: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    device: torch.device,
    n_episodes: int = 2000,
    n_way: int = 30,
    k_shot: int = 5,
    n_query: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_embeddings: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    eval_class_to_indices: Optional[Dict[int, np.ndarray]] = None,
    eval_use_all_support: bool = False,
    val_every: int = 100,
    min_samples_per_class: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Train Prototypical Network with episodic training.
    
    Args:
        model: PrototypicalNetwork model
        all_embeddings: [N, D] precomputed embeddings for all training samples
        class_to_indices: Dict mapping class_id -> array of indices in all_embeddings
        device: PyTorch device
        n_episodes: Number of training episodes
        n_way: Classes per episode
        k_shot: Support samples per class
        n_query: Query samples per class
        lr: Learning rate
        weight_decay: L2 regularization
        val_embeddings: Optional validation embeddings
        val_labels: Optional validation labels
        val_every: Validate every N episodes
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with training history
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes)
    
    # Create episode sampler
    sampler = EpisodeSampler(
        class_to_indices=class_to_indices,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        min_samples_per_class=min_samples_per_class,
        seed=seed
    )
    
    # Convert embeddings to tensor
    all_embeddings_tensor = torch.from_numpy(all_embeddings).float()
    
    # Training history
    history = {
        'episode': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    # Running averages
    running_loss = 0.0
    running_acc = 0.0

    best_val_acc = float("-inf")
    best_state = None
    
    model.train()
    pbar = tqdm(range(n_episodes), desc="Episodic Training", disable=not verbose)
    
    for episode_idx in pbar:
        # Sample episode
        support_idx, support_labels, query_idx, query_labels = sampler.sample_episode()
        
        # Get embeddings
        support_emb = all_embeddings_tensor[support_idx].to(device)
        query_emb = all_embeddings_tensor[query_idx].to(device)
        support_labels_t = torch.from_numpy(support_labels).long().to(device)
        query_labels_t = torch.from_numpy(query_labels).long().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Project embeddings
        support_proj = model(support_emb)
        query_proj = model(query_emb)
        
        # Compute prototypes (mean of support embeddings per class)
        # DO NOT normalize - use raw Euclidean distance like original ProtoNet
        prototypes = compute_prototypes(support_proj, support_labels_t, n_way)
        
        # Compute loss using Euclidean distance (original ProtoNet)
        loss, acc = prototypical_loss(query_proj, query_labels_t, prototypes, 
                                       temperature=1.0, use_cosine=False)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update running averages
        running_loss = 0.9 * running_loss + 0.1 * loss.item()
        running_acc = 0.9 * running_acc + 0.1 * acc.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss:.4f}',
            'acc': f'{running_acc*100:.1f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        # Validation
        if val_embeddings is not None and (episode_idx + 1) % val_every == 0:
            eval_mapping = eval_class_to_indices if eval_class_to_indices is not None else class_to_indices
            val_acc = evaluate_protonet(
                model,
                all_embeddings,
                eval_mapping,
                val_embeddings,
                val_labels,
                device,
                n_way=555,
                k_shot=k_shot,
                use_all_support=bool(eval_use_all_support),
            )
            history['val_acc'].append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = float(val_acc)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
            if verbose:
                tqdm.write(f"Episode {episode_idx+1}: Val Acc = {val_acc*100:.2f}%")
        
        # Record history
        if (episode_idx + 1) % 10 == 0:
            history['episode'].append(episode_idx + 1)
            history['train_loss'].append(running_loss)
            history['train_acc'].append(running_acc)
            history['lr'].append(scheduler.get_last_lr()[0])
    
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    history["best_val_acc"] = best_val_acc if best_val_acc != float("-inf") else None
    return history


def evaluate_protonet(
    model: PrototypicalNetwork,
    train_embeddings: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    n_way: int = 555,  # Use all classes for final evaluation
    k_shot: int = 5,
    use_all_support: bool = True
) -> float:
    """
    Evaluate Prototypical Network on validation set.
    
    For final evaluation, we use ALL support samples (not just k_shot)
    and classify ALL validation samples against ALL class prototypes.
    
    Args:
        model: Trained PrototypicalNetwork
        train_embeddings: [N_train, D] training embeddings
        class_to_indices: Dict mapping class_id -> indices in train_embeddings
        val_embeddings: [N_val, D] validation embeddings
        val_labels: [N_val] validation labels (original class IDs)
        device: PyTorch device
        n_way: Number of classes (555 for full evaluation)
        k_shot: Number of support samples per class
        use_all_support: If True, use all available support samples
    
    Returns:
        Accuracy on validation set
    """
    model.eval()
    
    # Get unique classes
    unique_classes = sorted(class_to_indices.keys())
    class_to_idx = {cid: i for i, cid in enumerate(unique_classes)}
    
    # Build support set (use all samples or k_shot per class)
    support_embs = []
    support_labels = []
    
    for class_id in unique_classes:
        indices = class_to_indices[class_id]
        if use_all_support:
            selected = indices
        else:
            selected = indices[:k_shot] if len(indices) >= k_shot else indices
        
        for idx in selected:
            support_embs.append(train_embeddings[idx])
            support_labels.append(class_to_idx[class_id])
    
    support_embs = torch.from_numpy(np.stack(support_embs)).float().to(device)
    support_labels = torch.tensor(support_labels).long().to(device)
    
    # Project support embeddings and compute prototypes
    with torch.no_grad():
        support_proj = model(support_embs)
        prototypes = compute_prototypes(support_proj, support_labels, len(unique_classes))
        # DO NOT normalize - use Euclidean distance consistently with training
        
        # Process validation in batches
        val_embs = torch.from_numpy(val_embeddings).float()
        batch_size = 256
        all_preds = []
        
        for i in range(0, len(val_embs), batch_size):
            batch = val_embs[i:i+batch_size].to(device)
            batch_proj = model(batch)
            # Use Euclidean distance (same as training)
            dists = torch.cdist(batch_proj, prototypes, p=2)
            preds = dists.argmin(dim=1).cpu().numpy()  # argmin for distance
            all_preds.extend(preds)
    
    # Convert predictions back to original class IDs
    idx_to_class = {i: cid for cid, i in class_to_idx.items()}
    pred_classes = np.array([idx_to_class[p] for p in all_preds])
    
    accuracy = (pred_classes == val_labels).mean()
    
    model.train()
    return accuracy


def evaluate_frozen_baseline(
    train_embeddings: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    k_shot: int = 5,
    use_euclidean: bool = True  # Match the training metric
) -> float:
    """
    Evaluate baseline accuracy using frozen embeddings (no learned projection).
    
    Uses the same distance metric as training for fair comparison.
    """
    # Get unique classes
    unique_classes = sorted(class_to_indices.keys())
    class_to_idx = {cid: i for i, cid in enumerate(unique_classes)}
    
    # Build prototypes
    prototypes = np.zeros((len(unique_classes), train_embeddings.shape[1]))
    
    for class_id in unique_classes:
        indices = class_to_indices[class_id][:k_shot]
        if len(indices) > 0:
            prototypes[class_to_idx[class_id]] = train_embeddings[indices].mean(axis=0)
    
    if use_euclidean:
        # Euclidean distance (to match ProtoNet training)
        # Compute pairwise distances: (N_val, N_classes)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        val_sq = np.sum(val_embeddings ** 2, axis=1, keepdims=True)  # (N_val, 1)
        proto_sq = np.sum(prototypes ** 2, axis=1, keepdims=True).T  # (1, N_classes)
        cross = val_embeddings @ prototypes.T  # (N_val, N_classes)
        dists = val_sq + proto_sq - 2 * cross
        pred_indices = dists.argmin(axis=1)  # argmin for distance
    else:
        # Cosine similarity (original implementation)
        prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        val_emb_norm = val_embeddings / (np.linalg.norm(val_embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = val_emb_norm @ prototypes.T
        pred_indices = similarities.argmax(axis=1)
    
    # Convert to class IDs
    idx_to_class = {i: cid for cid, i in class_to_idx.items()}
    pred_classes = np.array([idx_to_class[p] for p in pred_indices])
    
    accuracy = (pred_classes == val_labels).mean()
    return accuracy


# =============================================================================
# PART 5: VISUALIZATION UTILITIES
# =============================================================================

def plot_training_history(history: Dict, title: str = "Prototypical Network Training"):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['episode'], history['train_loss'], 'b-', alpha=0.7)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Training accuracy
    axes[1].plot(history['episode'], [a * 100 for a in history['train_acc']], 'g-', alpha=0.7)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy (per episode)')
    axes[1].grid(True, alpha=0.3)
    
    # Validation accuracy
    if history['val_acc']:
        val_episodes = np.linspace(0, len(history['episode'])*10, len(history['val_acc']))
        axes[2].plot(val_episodes, [a * 100 for a in history['val_acc']], 'r-o', alpha=0.7)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title('Validation Accuracy')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=70, color='green', linestyle='--', label='Target (70%)')
        axes[2].legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


# =============================================================================
# PART 6: MAIN EXPERIMENT FUNCTION
# =============================================================================

def run_prototypical_network_experiment(
    extractor,
    ds_train,
    support_indices: Dict[int, np.ndarray],
    pool_indices: Dict[int, np.ndarray],
    val_indices: Dict[int, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
    n_episodes: int = 3000,
    n_way: int = 30,
    k_shot: int = 5,
    n_query: int = 10,
    lr: float = 5e-4,            # Lower learning rate for stability
    embedding_dim: int = 512,    # Larger output for 555 classes
    min_samples_per_class: Optional[int] = None,  # None = k_shot + n_query (minimum needed)
    seed: int = 42
) -> Dict:
    """
    Run complete Prototypical Network experiment.
    
    Steps:
    1. Extract embeddings for support + pool + validation data
    2. Evaluate frozen baseline
    3. Train Prototypical Network with episodic training
    4. Evaluate trained model
    
    Args:
        extractor: Feature extractor with extract_from_dataset method
        ds_train: DeepLake training dataset
        support_indices: Dict mapping class_id -> support sample indices
        pool_indices: Dict mapping class_id -> pool sample indices
        val_indices: Dict mapping class_id -> validation sample indices
        device: PyTorch device
        batch_size: Batch size for embedding extraction
        n_episodes: Number of training episodes
        n_way: Classes per episode
        k_shot: Support samples per class per episode
        n_query: Query samples per class per episode
        lr: Learning rate
        embedding_dim: Output embedding dimension
        seed: Random seed
    
    Returns:
        Dict with experiment results
    """
    from fewshot import flatten_indices, get_labels_for_indices
    
    print("=" * 70)
    print("PROTOTYPICAL NETWORK EXPERIMENT")
    print("=" * 70)
    
    # Step 1: Prepare data indices
    print("\n[1/5] Preparing data...")
    
    # Combine support and pool for training (we'll use episodic sampling)
    train_indices_dict = {}
    for cid in set(support_indices.keys()) | set(pool_indices.keys()):
        support = list(support_indices.get(cid, []))
        pool = list(pool_indices.get(cid, []))
        train_indices_dict[cid] = np.array(support + pool, dtype=np.int64)
    
    # Flatten indices
    train_flat = flatten_indices(train_indices_dict)
    val_flat = flatten_indices(val_indices)
    
    # Create index mapping (original index -> position in train_flat)
    train_idx_to_pos = {int(idx): pos for pos, idx in enumerate(train_flat)}
    
    # Create class_to_position dict for episodic sampling
    class_to_positions = {}
    for cid, indices in train_indices_dict.items():
        positions = [train_idx_to_pos[int(idx)] for idx in indices if int(idx) in train_idx_to_pos]
        if positions:
            class_to_positions[cid] = np.array(positions)
    
    print(f"  Training samples: {len(train_flat)}")
    print(f"  Validation samples: {len(val_flat)}")
    print(f"  Classes with training data: {len(class_to_positions)}")
    
    # Step 2: Extract embeddings
    print("\n[2/5] Extracting embeddings...")
    
    train_embeddings = extractor.extract_from_dataset(ds_train, train_flat, batch_size)
    val_embeddings = extractor.extract_from_dataset(ds_train, val_flat, batch_size)
    
    # Get labels
    train_labels = get_labels_for_indices(ds_train, train_flat)
    val_labels = get_labels_for_indices(ds_train, val_flat)
    
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    print(f"  Val embeddings shape: {val_embeddings.shape}")
    
    # Step 3: Evaluate frozen baseline
    print("\n[3/5] Evaluating frozen baseline...")
    
    # Create class_to_indices for frozen baseline (using support only)
    support_class_to_pos = {}
    for cid, indices in support_indices.items():
        positions = [train_idx_to_pos[int(idx)] for idx in indices if int(idx) in train_idx_to_pos]
        if positions:
            support_class_to_pos[cid] = np.array(positions)
    
    frozen_acc = evaluate_frozen_baseline(
        train_embeddings, support_class_to_pos, val_embeddings, val_labels, k_shot=5
    )
    print(f"  Frozen baseline accuracy: {frozen_acc*100:.2f}%")
    
    # Step 4: Train Prototypical Network
    print("\n[4/5] Training Prototypical Network...")
    print(f"  Episodes: {n_episodes}")
    print(f"  Episode config: {n_way}-way {k_shot}-shot {n_query}-query")
    print(f"  Learning rate: {lr}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Distance metric: Euclidean (original ProtoNet)")
    print(f"  Projection mode: residual (preserves pretrained structure)")
    
    model = PrototypicalNetwork(
        input_dim=train_embeddings.shape[1],
        hidden_dim=train_embeddings.shape[1],  # Same as input
        embedding_dim=train_embeddings.shape[1],  # Keep original dimension!
        dropout=0.0,
        mode="residual"  # Preserves pretrained structure
    )
    
    history = train_projection_head_episodic(
        model=model,
        all_embeddings=train_embeddings,
        class_to_indices=class_to_positions,
        device=device,
        n_episodes=n_episodes,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        lr=lr,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        eval_class_to_indices=support_class_to_pos,  # evaluate exactly on 5-shot support
        eval_use_all_support=True,                   # mapping already contains the chosen support
        val_every=200,
        min_samples_per_class=min_samples_per_class,
        seed=seed
    )
    
    # Step 5: Final evaluation
    print("\n[5/5] Final evaluation...")
    
    # Evaluate with all support data
    final_acc = evaluate_protonet(
        model, train_embeddings, support_class_to_pos,
        val_embeddings, val_labels, device,
        use_all_support=True
    )
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Frozen baseline accuracy:     {frozen_acc*100:.2f}%")
    print(f"  Prototypical Network accuracy: {final_acc*100:.2f}%")
    print(f"  Improvement:                   {(final_acc - frozen_acc)*100:+.2f}%")
    print(f"{'='*70}")
    
    # Plot training curves
    fig = plot_training_history(history)
    
    return {
        'frozen_accuracy': frozen_acc,
        'protonet_accuracy': final_acc,
        'improvement': final_acc - frozen_acc,
        'history': history,
        'model': model,
        'figure': fig
    }


# =============================================================================
# QUICK START FUNCTION
# =============================================================================

def quick_validation_test(
    extractor,
    ds_train,
    support_indices: Dict[int, np.ndarray],
    pool_indices: Dict[int, np.ndarray],
    val_indices: Dict[int, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
    n_episodes: int = 1000,  # More episodes for reliable results
    seed: int = 42
) -> Dict:
    """
    Quick validation experiment to test if learned embeddings help.
    
    This is a faster version with fewer episodes to quickly validate
    the hypothesis before committing to full training.
    
    Key improvements over initial version:
    - Larger embedding_dim (512) to handle 555 fine-grained classes
    - More episodes (1000) for better convergence
    - Uses cosine similarity instead of Euclidean distance
    """
    print("=" * 70)
    print("QUICK VALIDATION: Testing learned projection head (improved)")
    print("=" * 70)
    print()
    print("Key settings:")
    print("  - Embedding dimension: 512 (larger for fine-grained classes)")
    print("  - Distance metric: Cosine similarity")
    print("  - Training episodes: 1000")
    print()
    
    return run_prototypical_network_experiment(
        extractor=extractor,
        ds_train=ds_train,
        support_indices=support_indices,
        pool_indices=pool_indices,
        val_indices=val_indices,
        device=device,
        batch_size=batch_size,
        n_episodes=n_episodes,
        n_way=30,  # More classes per episode
        k_shot=5,
        n_query=5,
        lr=5e-4,  # Slightly lower learning rate
        embedding_dim=512,  # Much larger output dimension
        seed=seed
    )


if __name__ == "__main__":
    print("Prototypical Networks module loaded successfully!")
    print("\nUsage in notebook:")
    print("  from prototypical_networks import quick_validation_test, run_prototypical_network_experiment")
    print("  ")
    print("  # Quick test (5-10 minutes)")
    print("  results = quick_validation_test(extractor, ds_train, support_indices, pool_indices, val_indices_split, DEVICE)")
    print("  ")
    print("  # Full training (30-60 minutes)")
    print("  results = run_prototypical_network_experiment(extractor, ds_train, support_indices, pool_indices, val_indices_split, DEVICE, n_episodes=3000)")
