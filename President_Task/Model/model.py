import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ==========================================
# 1. MODÈLES PYTORCH CORRIGÉS
# ==========================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLP(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1) # Sortie : Logits
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

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
        # x shape: (batch_size, seq_len, vector_size)
        rnn_out, _ = self.rnn(x)

        # CORRECTION : On applique la couche linéaire sur CHAQUE phrase de la séquence
        # out shape: (batch_size, seq_len)
        out = self.fc(self.dropout(rnn_out)).squeeze(-1)

        # On masque les éléments "paddés" (remplacés par -inf pour le BCEWithLogitsLoss)
        if lengths is not None:
            mask = torch.arange(out.size(1), device=x.device)[None, :] >= lengths[:, None]
            out = out.masked_fill(mask, float('-inf'))
            
        return out

# ==========================================
# 2. WRAPPER 
# ==========================================

class TorchTrainer:
    """
    Wrapper universel pour entraîner MLP ou RNN comme avec Scikit-Learn,
    avec support optionnel de la validation (X_val, y_val).
    """
    def __init__(self, model, epochs=10, batch_size=32, lr=1e-3, pos_weight=6.6, device='cpu'):
        self.model = model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        # Le pos_weight intègre directement ton facteur 6.6
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y, X_val=None, y_val=None):
        # 1. Préparation des données d'entraînement
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 2. Préparation des données de validation (si fournies)
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            # Pas besoin de shuffle pour la validation
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 3. Boucle d'entraînement
        for epoch in range(self.epochs):
            # --- PHASE D'ENTRAÎNEMENT ---
            self.model.train()
            total_train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                
                # Ignorer les labels -1 (Padding) si on utilise des séquences
                mask = (batch_y != -1)
                loss = self.criterion(logits[mask], batch_y[mask])
                
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            
            # --- PHASE DE VALIDATION ---
            if has_val:
                self.model.eval()
                total_val_loss = 0
                
                with torch.no_grad(): # Désactive le calcul des gradients pour aller plus vite
                    for batch_X_val, batch_y_val in val_loader:
                        batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                        
                        logits_val = self.model(batch_X_val)
                        mask_val = (batch_y_val != -1)
                        val_loss = self.criterion(logits_val[mask_val], batch_y_val[mask_val])
                        
                        total_val_loss += val_loss.item()
                        
                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch {epoch+1:02d}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                # Si pas de validation, on affiche juste la Train Loss
                print(f"Epoch {epoch+1:02d}/{self.epochs} | Train Loss: {avg_train_loss:.4f}")

    def predict_proba(self, X):
        """Retourne de vraies probabilités (entre 0 et 1) grâce à la Sigmoïde"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # On peut aussi utiliser un DataLoader ici si X est trop gros,
        # mais on va supposer que ça passe en un seul bloc pour le moment.
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)
            
        return probs.cpu().numpy()

# ==========================================
# 3. LES FONCTIONS HMM (Post-Processing)
# ==========================================

def compute_real_transitions(df, target_col='target', doc_col='Doc_ID'):
    """
    Calcule la matrice de transition A (2x2) à partir des vrais labels.
    A[i, j] = P(état_suivant = j | état_actuel = i)
    """
    # Initialisation de la matrice des occurrences (2 classes : 0 et 1)
    transition_counts = np.zeros((2, 2))
    
    # On s'assure que les données sont bien dans l'ordre chronologique
    df_sorted = df.sort_values([doc_col, 'Sentence_ID'])
    
    # On parcourt chaque débat indépendamment
    for doc_id, group in df_sorted.groupby(doc_col):
        labels = group[target_col].values
        
        # On compte les transitions d'une phrase à l'autre
        for t in range(len(labels) - 1):
            etat_actuel = int(labels[t])
            etat_suivant = int(labels[t + 1])
            
            # Sécurité : on s'assure que ce sont bien des 0 ou des 1
            if etat_actuel in [0, 1] and etat_suivant in [0, 1]:
                transition_counts[etat_actuel, etat_suivant] += 1
                
    # Normalisation : on transforme les comptes en probabilités (la somme de chaque ligne = 1)
    # On ajoute un minuscule lissage (1e-9) pour éviter les divisions par zéro si une transition n'existe pas
    A_real = (transition_counts + 1e-9) / (transition_counts.sum(axis=1, keepdims=True) + 2e-9)
    
    return A_real

def viterbi_hmm(probs_c0, trans_matrix, pi, weight_c0=6.6):
    n = len(probs_c0)
    
    # Passage en log pour la stabilité (évite les nombres trop petits)
    log_pi = np.log(pi + 1e-12)
    log_trans = np.log(trans_matrix + 1e-12)
    
    # Émissions : on booste la classe 0 avec ton facteur 6.6
    # log(p * weight) = log(p) + log(weight)
    emissions = np.vstack([probs_c0, 1 - probs_c0])
    log_emissions = np.log(emissions + 1e-12)
    log_emissions[0, :] += np.log(weight_c0) 

    viterbi_table = np.zeros((2, n))
    backpointer = np.zeros((2, n), dtype=int)

    # Initialisation (t=0)
    viterbi_table[:, 0] = log_pi + log_emissions[:, 0]

    # Forward pass
    for t in range(1, n):
        for s in range(2):
            # On cherche le max de (score précédent + transition + émission actuelle)
            scores = viterbi_table[:, t-1] + log_trans[:, s] + log_emissions[s, t]
            viterbi_table[s, t] = np.max(scores)
            backpointer[s, t] = np.argmax(scores)

    # Backward pass (Backtracking)
    path = np.zeros(n, dtype=int)
    path[n-1] = np.argmax(viterbi_table[:, n-1])
    for t in range(n-2, -1, -1):
        path[t] = backpointer[path[t+1], t+1]
        
    return path

# 2. L'ORCHESTRE : Application par segments Doc_ID
def apply_viterbi_segmented(df, matrix_A, weight_c0=6.6):
    # On identifie les segments continus pour ne pas lisser entre deux débats
    df['is_break'] = (df['Doc_ID'] != df['Doc_ID'].shift()) | \
                     (df['Sentence_ID'] != df['Sentence_ID'].shift() + 1)
    df['segment_id'] = df['is_break'].cumsum()
    
    final_labels = []
    pi_global = np.array([0.13, 0.87]) # Proba de départ moyenne

    for _, segment in df.groupby('segment_id'):
        probs = segment['Prob_Mitterrand'].values
        
        if len(segment) < 2:
            # Si phrase isolée, on applique juste le seuil pondéré
            seuil = 1 / (weight_c0 + 1)
            preds = (probs > seuil).astype(int)
        else:
            # Application du HMM sur le segment continu
            preds = viterbi_hmm(probs, matrix_A, pi_global, weight_c0)
            
        final_labels.extend(preds)
    
    return np.array(final_labels)

def forward_backward_smoother(probs_c0, trans_matrix, pi, weight_c0=6.6):
    n = len(probs_c0)
    # On prépare les émissions (avec ton poids 6.6)
    # Attention : on reste en espace linéaire (pas log) pour les sommes, 
    # mais on utilise un facteur d'échelle pour éviter l'underflow.
    obs = np.vstack([probs_c0 * weight_c0, 1 - probs_c0])
    obs /= obs.sum(axis=0) # Normalisation locale des émissions

    # Forward Pass (Alpha)
    alpha = np.zeros((2, n))
    scale = np.zeros(n)
    
    alpha[:, 0] = pi * obs[:, 0]
    scale[0] = alpha[:, 0].sum()
    alpha[:, 0] /= scale[0]
    
    for t in range(1, n):
        alpha[:, t] = (alpha[:, t-1] @ trans_matrix) * obs[:, t]
        scale[t] = alpha[:, t].sum()
        alpha[:, t] /= scale[t]

    # Backward Pass (Beta)
    beta = np.zeros((2, n))
    beta[:, n-1] = 1.0
    
    for t in range(n-2, -1, -1):
        beta[:, t] = (trans_matrix @ (obs[:, t+1] * beta[:, t+1]))
        beta[:, t] /= scale[t+1] # On utilise le même scale pour la stabilité

    # Calcul des probabilités postérieures (Gamma)
    posterior = alpha * beta
    posterior /= posterior.sum(axis=0)
    
    return posterior[0, :] # On renvoie uniquement la proba lissée de la classe 0

# --- Wrapper pour les segments ---
def apply_hmm_proba_segmented(df, matrix_A, weight_c0=6.6):
    df['is_break'] = (df['Doc_ID'] != df['Doc_ID'].shift()) | \
                     (df['Sentence_ID'] != df['Sentence_ID'].shift() + 1)
    df['segment_id'] = df['is_break'].cumsum()
    
    final_probs = []
    pi_global = np.array([0.13, 0.87])

    for _, segment in df.groupby('segment_id'):
        p_raw = segment['Prob_Mitterrand'].values
        
        if len(p_raw) < 2:
            # Pour une phrase seule, on applique juste le boost du poids
            p_boosted = (p_raw * weight_c0) / (p_raw * weight_c0 + (1 - p_raw))
            final_probs.extend(p_boosted)
        else:
            # Lissage Forward-Backward
            p_smooth = forward_backward_smoother(p_raw, matrix_A, pi_global, weight_c0)
            final_probs.extend(p_smooth)
            
    return np.array(final_probs)