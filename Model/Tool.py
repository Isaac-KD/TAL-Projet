import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler=None,
    epochs=10,
    device="cpu",
    scheduler_step="batch",  # "batch" ou "epoch"
    verbose=True,
):
    """
    Fonction d'entraînement générique avec suivi de la loss train/val.

    Args:
        model         : Le modèle PyTorch à entraîner.
        train_loader  : DataLoader d'entraînement.
        val_loader    : DataLoader de validation.
        optimizer     : Optimiseur (Adam, SGD, etc.).
        criterion     : Fonction de loss.
        scheduler     : (Optionnel) Learning rate scheduler.
        epochs        : Nombre d'époques.
        device        : "cpu" ou "cuda".
        scheduler_step: "batch" pour OneCycleLR, "epoch" pour ReduceLROnPlateau etc.
        verbose       : Affiche les logs si True.

    Returns:
        history (dict): {"train_loss": [...], "val_loss": [...]}
    """
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    if verbose:
        print(f"🚀 Début de l'entraînement sur {device} pour {epochs} époques...")

    for epoch in range(epochs):

        # ── ENTRAÎNEMENT ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if scheduler and scheduler_step == "batch":
                scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ── VALIDATION ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_outputs, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                val_loss += criterion(outputs, targets).item()

                preds = outputs.cpu().numpy()
                trues = targets.cpu().numpy()

                all_outputs.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
                all_targets.extend(trues.tolist() if trues.ndim > 0 else [trues.item()])

        avg_val_loss = val_loss / len(val_loader)

        if scheduler and scheduler_step == "epoch":
            # ReduceLROnPlateau attend la val_loss en argument
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # ── HISTORIQUE & LOGS ─────────────────────────────────────────────────
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if verbose:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1:>3}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"LR: {lr:.2e}"
            )

    return history, all_outputs, all_targets

def evaluate(
    model,
    val_loader,
    device="cpu",
    run_name="Experiment",
    threshold=0.5,
    verbose=True,
):
    """
    Évalue un modèle de classification binaire sur un DataLoader.

    Args:
        model     : Modèle PyTorch entraîné.
        val_loader: DataLoader de validation.
        device    : "cpu" ou "cuda".
        run_name  : Nom de l'expérience (pour la colonne 'Run').
        threshold : Seuil de décision pour les classes (défaut 0.5).
        verbose   : Affiche les métriques si True.

    Returns:
        metriques (dict): Dictionnaire de toutes les métriques.
    """
    model.to(device)
    model.eval()

    all_probs, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Gestion sigmoid si le modèle ne l'applique pas en sortie
            probs = torch.sigmoid(outputs).squeeze()

            all_probs.extend(probs.cpu().numpy().tolist() if probs.dim() > 0 else [probs.item()])
            all_targets.extend(targets.cpu().numpy().tolist() if targets.dim() > 0 else [targets.item()])

    all_probs   = np.array(all_probs)
    all_targets = np.array(all_targets)
    preds_classes = (all_probs >= threshold).astype(int)

    metriques = {
        "Run"      : run_name,
        "Accuracy" : accuracy_score(all_targets, preds_classes),
        "Precision": precision_score(all_targets, preds_classes, zero_division=0),
        "Recall"   : recall_score(all_targets, preds_classes, zero_division=0),
        "F1_Score" : f1_score(all_targets, preds_classes, zero_division=0),
        "ROC_AUC"  : roc_auc_score(all_targets, all_probs),
    }

    if verbose:
        print("\n📊 Métriques finales sur le set de validation :")
        for k, v in metriques.items():
            print(f"  - {k} : {v:.4f}" if isinstance(v, float) else f"  - {k} : {v}")

    return metriques