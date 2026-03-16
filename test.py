import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Data.Tool import *

# ──────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────
VECTOR_SIZE  = 300     # dimension des embeddings FastText
HIDDEN_DIM   = 128
BIDIRECTIONAL= True
DROPOUT      = 0.3
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────
# 2. MODÈLE
# ──────────────────────────────────────────────
class RNN(nn.Module):
    def __init__(self, vector_size, hidden_dim, bi=False, dropout=0.3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=vector_size,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bi,
            num_layers=2,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (1 + bi), 1)

    def forward(self, x, lengths=None):
        rnn_out, _ = self.rnn(x)
        if lengths is not None:
            mask = (
                torch.arange(rnn_out.size(1), device=x.device)[None, :]
                >= lengths[:, None]
            )
            rnn_out = rnn_out.masked_fill(mask.unsqueeze(-1), float('-inf'))
        pooled = torch.max(rnn_out, dim=1)[0]
        return self.fc(self.dropout(pooled)).squeeze(-1)


# ──────────────────────────────────────────────
# 3. DATASET
# ──────────────────────────────────────────────
class SentenceDataset(Dataset):
    """
    Chaque item : (tensor de shape [T, vector_size], label 0/1)
    La colonne 'Sequence' peut être :
      - déjà une liste de listes (après pd.read_pickle / pd.read_parquet)
      - une chaîne de caractères à parser (après pd.read_csv)
    La colonne 'Label' contient 'C' ou 'M'.
    """
    def __init__(self, df: pd.DataFrame):
        self.sequences = []
        self.labels    = []

        for _, row in df.iterrows():
            seq = row["Sequence"]
            # Si la séquence est stockée sous forme de string (CSV)
            if isinstance(seq, str):
                seq = ast.literal_eval(seq)
            tensor = torch.tensor(seq, dtype=torch.float32)  # [T, D]
            self.sequences.append(tensor)
            self.labels.append(1.0 if row["Label"] == "C" else 0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], torch.tensor(self.labels[idx])


def collate_fn(batch):
    """Padding des séquences de longueurs variables + longueurs réelles."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs])
    padded  = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels  = torch.stack(labels)
    return padded, lengths, labels


# ──────────────────────────────────────────────
# 4. CHARGEMENT DES DONNÉES
# ──────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """
    Accepte .csv, .pkl, .parquet.
    Adaptez le séparateur CSV si nécessaire.
    """
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Format non reconnu : {path}")


# ──────────────────────────────────────────────
# 5. BOUCLES TRAIN / EVAL
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, lengths, y in loader:
        X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X, lengths)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total   += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, lengths, y in loader:
        X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
        logits = model(X, lengths)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total   += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# 6. VISUALISATIONS
# ──────────────────────────────────────────────
def plot_curves(history: dict, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["loss", "acc"]):
        ax.plot(history[f"train_{metric}"], label="Train", marker="o")
        ax.plot(history[f"val_{metric}"],   label="Val",   marker="s")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Courbes sauvegardées → {save_path}")


def plot_confusion(labels, preds, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["M", "C"], yticklabels=["M", "C"], ax=ax
    )
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Matrice de confusion sauvegardée → {save_path}")

# ──────────────────────────────────────────────
# 7. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────
def main(data_path: str):
    # --- Données ---
    df = load_and_prepare(
        corpus_path = "Data/train/corpus.tache1.learn.utf8",
        pkl_path    = "Data/train/sequences_fasttext_fr.pkl",
        vector_size = 300,
    )
    print(f"Dataset chargé : {len(df)} lignes | Labels : {df['Label'].value_counts().to_dict()}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["Label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=SEED, stratify=train_df["Label"]
    )
    print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

    train_loader = DataLoader(
        SentenceDataset(train_df), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader  = DataLoader(
        SentenceDataset(val_df), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SentenceDataset(test_df), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    # --- Modèle ---
    model     = RNN(VECTOR_SIZE, HIDDEN_DIM, bi=BIDIRECTIONAL, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # Gestion du déséquilibre de classes
    n_C = (train_df["Label"] == "C").sum()
    n_M = (train_df["Label"] == "M").sum()
    pos_weight = torch.tensor([n_M / n_C], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Entraînement ---
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 5

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc          = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, _, _    = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
            f"Val Loss={vl_loss:.4f} Acc={vl_acc:.4f}"
        )

        # Sauvegarde du meilleur modèle
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping à l'époque {epoch}.")
                break

    # --- Évaluation finale ---
    model.load_state_dict(torch.load("best_model.pt"))
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion)

    # Convertir les prédictions en labels texte
    label_map  = {1.0: "C", 0.0: "M"}
    pred_names = [label_map[p] for p in test_preds]
    true_names = [label_map[l] for l in test_labels]

    print(f"\n=== Résultats sur le jeu de test ===")
    print(f"Accuracy : {test_acc:.4f}\n")
    print(classification_report(true_names, pred_names, target_names=["M", "C"]))

    plot_curves(history)
    plot_confusion(true_names, pred_names)
    print("\nFichiers générés : training_curves.png | confusion_matrix.png | best_model.pt")


# ──────────────────────────────────────────────
# 8. INFÉRENCE SUR DE NOUVELLES DONNÉES
# ──────────────────────────────────────────────
@torch.no_grad()
def predict(model, sequences: list[list]) -> list[dict]:
    """
    sequences : liste de matrices (liste de vecteurs)
    Retourne : liste de dict {'label': 'C'/'M', 'proba': float}
    """
    model.eval()
    results = []
    for seq in sequences:
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        logit = model(x)
        proba = torch.sigmoid(logit).item()
        results.append({"label": "C" if proba > 0.5 else "M", "proba": round(proba, 4)})
    return results


# ──────────────────────────────────────────────
#if __name__ == "__main__":
    #import sys
    #data_path = sys.argv[1] if len(sys.argv) > 1 else "Data/train/.pkl"
    #main(data_path)