# =============================================================================
# COMPLETE LEARNED PROJECTION MODULE (Paste this in Colab)
# =============================================================================
def get_embeddings_fast(indices):
    """Instant lookup from pre-computed embeddings (O(n) instead of minutes)."""
    positions = [_idx_to_pos[int(i)] for i in indices]
    return ALL_EMBEDDINGS[positions]



def flatten_class_indices(indices_dict):
    """Flatten a dict of {class_id: [indices]} to a list of (index, class_id) tuples."""
    flat_indices = []
    flat_labels = []
    for class_id, indices in indices_dict.items():
        for idx in indices:
            flat_indices.append(int(idx))
            flat_labels.append(int(class_id))
    return flat_indices, flat_labels



def flatten_class_indices_local(indices_dict):
    flat_indices = []
    flat_labels = []
    for class_id, indices in indices_dict.items():
        for idx in indices:
            flat_indices.append(int(idx))
            flat_labels.append(int(class_id))
    return flat_indices, flat_labels


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=1024, output_dim=512, dropout=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        if use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        z = self.projection(x)
        if self.use_residual:
            residual = self.residual_proj(x) if self.residual_proj else x
            z = z + 0.1 * residual
        return F.normalize(z, p=2, dim=1)


class SupervisedProjectionClassifier(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=1024, embedding_dim=512, n_classes=555, dropout=0.1):
        super().__init__()
        self.projection = ProjectionHead(input_dim, hidden_dim, embedding_dim, dropout, True)
        self.classifier = nn.Linear(embedding_dim, n_classes, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        z = self.projection(x)
        w = F.normalize(self.classifier.weight, p=2, dim=1)
        return self.scale * torch.mm(z, w.t())

    def get_embedding(self, x):
        return self.projection(x)


def evaluate_with_prototypes(model, train_embeddings, train_labels, val_embeddings, val_labels, device, return_predictions=False, debug=False):
    model.eval()
    train_labels = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    val_labels = np.asarray(val_labels, dtype=np.int64).reshape(-1)

    unique_classes = np.unique(train_labels)
    n_classes = len(unique_classes)
    class_to_idx = {int(c): i for i, c in enumerate(unique_classes)}

    if debug:
        print(f"DEBUG: n_classes={n_classes}, train_labels sample: {train_labels[:5]}, val_labels sample: {val_labels[:5]}")

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
            batch = val_emb_t[i:i+256].to(device)
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


def evaluate_frozen_baseline(train_embeddings, train_labels, val_embeddings, val_labels):
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
    val_norm = val_embeddings / (np.linalg.norm(val_embeddings, axis=1, keepdims=True) + 1e-8)

    similarities = val_norm @ prototypes.T
    pred_indices = similarities.argmax(axis=1)

    idx_to_class = {i: c for c, i in class_to_idx.items()}
    pred_labels = np.array([idx_to_class[int(p)] for p in pred_indices], dtype=np.int64)

    return (pred_labels == val_labels).mean()


def train_supervised_projection(
    model, train_embeddings, train_labels, train_labels_original,
    val_embeddings, val_labels, device,
    n_epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4,
    label_smoothing=0.1, verbose=True
):
    model = model.to(device)
    train_labels_original = np.asarray(train_labels_original, dtype=np.int64).reshape(-1)
    if val_labels is not None:
        val_labels = np.asarray(val_labels, dtype=np.int64).reshape(-1)

    train_dataset = TensorDataset(
        torch.from_numpy(train_embeddings).float(),
        torch.from_numpy(train_labels).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_acc': []}
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

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate every 5 epochs using ORIGINAL labels
        if val_embeddings is not None and (epoch + 1) % 5 == 0:
            val_acc = evaluate_with_prototypes(
                model, train_embeddings, train_labels_original,
                val_embeddings, val_labels, device
            )
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'train': f'{train_acc*100:.1f}%', 'val': f'{val_acc*100:.1f}%'})
        else:
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'train': f'{train_acc*100:.1f}%'})

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model with val accuracy = {best_val_acc*100:.2f}%")

    return history


def run_learned_projection_experiment(
    extractor, ds_train, support_indices, pool_indices, val_indices, device,
    batch_size=64, n_epochs=50, lr=1e-3, embedding_dim=512, use_pool_data=False, seed=42,
    embedding_lookup_fn=None  # Pass get_embeddings_fast for instant lookup
):
    from fewshot import flatten_indices, get_labels_for_indices

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
        train_embeddings = extractor.extract_from_dataset(ds_train, train_flat, batch_size)
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
    frozen_acc = evaluate_frozen_baseline(train_embeddings, train_labels, val_embeddings, val_labels)
    print(f"  Frozen accuracy: {frozen_acc*100:.2f}%")

    # Train
    print("\n[4/5] Training projection...")
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
        train_labels=train_labels_mapped,          # Mapped 0..N-1 for cross-entropy
        train_labels_original=train_labels,        # Original for prototype eval
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        device=device,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True
    )

    # Final eval
    print("\n[5/5] Final evaluation...")
    final_acc, pred_labels = evaluate_with_prototypes(
        model, train_embeddings, train_labels,
        val_embeddings, val_labels, device,
        return_predictions=True
    )

    val_precision = precision_score(val_labels, pred_labels, average='macro', zero_division=0)
    val_recall = recall_score(val_labels, pred_labels, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, pred_labels, average='macro', zero_division=0)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Validation Accuracy: {final_acc*100:.2f}%")
    print(f"  Precision: {val_precision*100:.2f}%  Recall: {val_recall*100:.2f}%  F1: {val_f1*100:.2f}%")
    print(f"{'='*70}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['epoch'], history['train_loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['epoch'], [a*100 for a in history['train_acc']], 'b-', alpha=0.7, label='Train')
    if history['val_acc']:
        val_epochs = list(range(5, n_epochs + 1, 5))[:len(history['val_acc'])]
        axes[1].plot(val_epochs, [a*100 for a in history['val_acc']], 'ro-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'frozen_accuracy': frozen_acc,
        'val_accuracy': final_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'history': history,
        'model': model
    }

print("✅ Complete learned_projection module loaded!")