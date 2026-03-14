"""
transformer_model.py
ZombieGuard Phase 2 - Byte-level Transformer classifier.

Architecture:
  - Input: first 512 raw bytes of ZIP file (byte values 0-255 as tokens)
  - Embedding: 256 vocab -> 64-dim embeddings
  - Positional encoding: learned
  - Transformer encoder: 2 layers, 4 heads, 128 hidden dim
  - Classifier head: 128 -> 64 -> 1 (sigmoid)

Rationale:
  This mirrors the approach in CrowdStrike's Binary Transformer research.
  A small model is appropriate for a 2647-sample dataset - large pre-trained
  models would overfit immediately on this data size.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- Config -------------------------------------------------
SEQ_LEN = 512
VOCAB_SIZE = 257
PAD_TOKEN = 256
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
N_SPLITS = 5

MALICIOUS_DIR = "data/raw/malicious"
BENIGN_DIR = "data/raw/benign"
MODEL_SAVE_PATH = "models/transformer_model.pt"


# -- Dataset -----------------------------------------------
class ZipByteDataset(Dataset):
	"""
	Reads raw bytes from ZIP files.
	Returns first SEQ_LEN bytes as token IDs.
	Pads shorter files with PAD_TOKEN.
	"""

	def __init__(self, file_paths: list, labels: list, seq_len: int = SEQ_LEN):
		self.file_paths = file_paths
		self.labels = labels
		self.seq_len = seq_len

	def __len__(self):
		"""Return dataset size."""
		return len(self.file_paths)

	def __getitem__(self, idx):
		"""Load one file and return padded byte tokens with binary label."""
		path = self.file_paths[idx]
		label = self.labels[idx]

		try:
			with open(path, "rb") as file:
				raw = file.read(self.seq_len)
			byte_ids = list(raw) + [PAD_TOKEN] * (self.seq_len - len(raw))
		except Exception:
			byte_ids = [PAD_TOKEN] * self.seq_len

		return (
			torch.tensor(byte_ids[: self.seq_len], dtype=torch.long),
			torch.tensor(label, dtype=torch.float),
		)


# -- Model -------------------------------------------------
class ByteTransformerClassifier(nn.Module):
	"""
	Small transformer encoder that classifies byte sequences
	as malicious (Zombie ZIP) or benign.
	"""

	def __init__(
		self,
		vocab_size: int = VOCAB_SIZE,
		embed_dim: int = EMBED_DIM,
		num_heads: int = NUM_HEADS,
		num_layers: int = NUM_LAYERS,
		ff_dim: int = FF_DIM,
		seq_len: int = SEQ_LEN,
		dropout: float = DROPOUT,
	):
		super().__init__()

		self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
		self.pos_embedding = nn.Embedding(seq_len, embed_dim)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model=embed_dim,
			nhead=num_heads,
			dim_feedforward=ff_dim,
			dropout=dropout,
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		# Classification head
		self.classifier = nn.Sequential(
			nn.Linear(embed_dim, 64),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(64, 1),
			nn.Sigmoid(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Run forward pass for a batch of byte-token sequences."""
		# x shape: (batch, seq_len)
		positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
		embedded = self.embedding(x) + self.pos_embedding(positions)

		# Transformer encoder
		encoded = self.transformer(embedded)

		# Global average pooling over sequence
		pooled = encoded.mean(dim=1)

		return self.classifier(pooled).squeeze(-1)


# -- Training helpers --------------------------------------
def _compute_metrics(y_true, y_pred, y_prob) -> dict:
	"""Compute binary classification metrics for Transformer evaluation."""
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"roc_auc": float(roc_auc_score(y_true, y_prob)),
	}


def _train_epoch(model, loader, optimizer, criterion, device):
	"""Train model for one epoch and return average batch loss."""
	model.train()
	total_loss = 0.0
	for x_batch, y_batch in loader:
		x_batch, y_batch = x_batch.to(device), y_batch.to(device)
		optimizer.zero_grad()
		preds = model(x_batch)
		loss = criterion(preds, y_batch)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	return total_loss / len(loader)


def _evaluate(model, loader, criterion, device):
	"""Evaluate model and return metrics, probabilities, and labels."""
	model.eval()
	all_probs, all_labels = [], []
	total_loss = 0.0
	with torch.no_grad():
		for x_batch, y_batch in loader:
			x_batch, y_batch = x_batch.to(device), y_batch.to(device)
			probs = model(x_batch)
			loss = criterion(probs, y_batch)
			total_loss += loss.item()
			all_probs.extend(probs.cpu().numpy())
			all_labels.extend(y_batch.cpu().numpy())

	all_probs = np.array(all_probs)
	all_labels = np.array(all_labels)
	all_preds = (all_probs >= 0.5).astype(int)
	metrics = _compute_metrics(all_labels, all_preds, all_probs)
	metrics["loss"] = total_loss / len(loader)
	return metrics, all_probs, all_labels


# -- Data loader -------------------------------------------
def load_file_paths_and_labels() -> tuple:
	"""Collect all file paths and their labels."""
	paths, labels = [], []

	for fname in os.listdir(MALICIOUS_DIR):
		if fname.endswith(".zip"):
			paths.append(os.path.join(MALICIOUS_DIR, fname))
			labels.append(1)

	for fname in os.listdir(BENIGN_DIR):
		if fname.endswith(".zip"):
			paths.append(os.path.join(BENIGN_DIR, fname))
			labels.append(0)

	return paths, labels


# -- Main training -----------------------------------------
def train_transformer():
	"""Train and evaluate the byte-level Transformer with CV and holdout split."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	# Load data
	paths, labels = load_file_paths_and_labels()
	paths = np.array(paths)
	labels = np.array(labels)

	print(f"Dataset: {len(paths)} files")
	print(f"  Malicious: {labels.sum()}")
	print(f"  Benign:    {(labels == 0).sum()}\n")

	# Holdout split
	idx = np.arange(len(paths))
	train_idx, test_idx = train_test_split(
		idx,
		test_size=0.2,
		stratify=labels,
		random_state=42,
	)

	# -- 5-fold cross-validation
	print(f"Running {N_SPLITS}-fold cross-validation...")
	skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
	fold_metrics = []

	for fold, (tr_idx, val_idx) in enumerate(skf.split(train_idx, labels[train_idx]), start=1):
		tr_paths = paths[train_idx[tr_idx]]
		tr_labels = labels[train_idx[tr_idx]].tolist()
		val_paths = paths[train_idx[val_idx]]
		val_labels = labels[train_idx[val_idx]].tolist()

		tr_loader = DataLoader(
			ZipByteDataset(tr_paths.tolist(), tr_labels),
			batch_size=BATCH_SIZE,
			shuffle=True,
		)
		val_loader = DataLoader(
			ZipByteDataset(val_paths.tolist(), val_labels),
			batch_size=BATCH_SIZE,
		)

		model = ByteTransformerClassifier().to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
		criterion = nn.BCELoss()
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

		best_val_f1 = 0.0
		for epoch in range(1, EPOCHS + 1):
			train_loss = _train_epoch(model, tr_loader, optimizer, criterion, device)
			val_metrics, _, _ = _evaluate(model, val_loader, criterion, device)
			scheduler.step()

			if epoch % 10 == 0:
				print(
					f"  Fold {fold} Epoch {epoch:02d}: "
					f"loss={train_loss:.4f}  "
					f"val_f1={val_metrics['f1']:.4f}  "
					f"val_rec={val_metrics['recall']:.4f}"
				)

			if val_metrics["f1"] > best_val_f1:
				best_val_f1 = val_metrics["f1"]

		val_metrics, _, _ = _evaluate(model, val_loader, criterion, device)
		val_metrics["fold"] = float(fold)
		fold_metrics.append(val_metrics)
		print(
			f"  Fold {fold} final: "
			f"acc={val_metrics['accuracy']:.4f}  "
			f"rec={val_metrics['recall']:.4f}  "
			f"f1={val_metrics['f1']:.4f}  "
			f"auc={val_metrics['roc_auc']:.4f}\n"
		)

	cv_df = pd.DataFrame(fold_metrics)
	cv_summary = cv_df.drop(columns=["fold", "loss"]).mean().to_dict()
	print(
		f"CV mean - "
		f"acc={cv_summary['accuracy']:.4f}  "
		f"rec={cv_summary['recall']:.4f}  "
		f"f1={cv_summary['f1']:.4f}  "
		f"auc={cv_summary['roc_auc']:.4f}"
	)

	# -- Final model on full training set
	print("\nTraining final model on full training set...")
	tr_paths = paths[train_idx]
	tr_labels = labels[train_idx].tolist()
	te_paths = paths[test_idx]
	te_labels = labels[test_idx].tolist()

	final_loader = DataLoader(
		ZipByteDataset(tr_paths.tolist(), tr_labels),
		batch_size=BATCH_SIZE,
		shuffle=True,
	)
	test_loader = DataLoader(
		ZipByteDataset(te_paths, te_labels),
		batch_size=BATCH_SIZE,
	)

	final_model = ByteTransformerClassifier().to(device)
	optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
	criterion = nn.BCELoss()
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

	for epoch in range(1, EPOCHS + 1):
		loss = _train_epoch(final_model, final_loader, optimizer, criterion, device)
		scheduler.step()
		if epoch % 10 == 0:
			print(f"  Epoch {epoch:02d} loss={loss:.4f}")

	# -- Holdout test evaluation
	test_metrics, test_probs, test_labels = _evaluate(final_model, test_loader, criterion, device)
	test_preds = (test_probs >= 0.5).astype(int)

	print("\n-- Transformer Holdout Test Results ----------------")
	print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
	print(f"  Precision : {test_metrics['precision']:.4f}")
	print(f"  Recall    : {test_metrics['recall']:.4f}")
	print(f"  F1        : {test_metrics['f1']:.4f}")
	print(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")
	print("\n-- Classification Report ----------------------------")
	print(classification_report(test_labels, test_preds, target_names=["Benign", "Malicious"]))
	print("-- Confusion Matrix ---------------------------------")
	cm = confusion_matrix(test_labels, test_preds)
	print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
	print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
	print("------------------------------------------------------")

	# Save model
	os.makedirs(Path(MODEL_SAVE_PATH).parent, exist_ok=True)
	torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
	print(f"\nTransformer model saved to: {MODEL_SAVE_PATH}")

	return {
		"cv_summary": cv_summary,
		"test_metrics": test_metrics,
		"model": final_model,
		"cv_fold_metrics": cv_df,
	}


if __name__ == "__main__":
	results = train_transformer()
	print("\nStep 11 complete.")
