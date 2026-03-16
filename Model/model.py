import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLP(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super(MLP, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5), # Régularisation forte car l'entrée est très grande
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1) # Sortie brute (Logit) pour la classification binaire
        )

    def forward(self, x):
        # x shape: (batch_size, vocab_size) - Vecteurs TF-IDF denses
        logits = self.net(x)
        return logits.squeeze(1) # shape: (batch_size,)

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
            mask = torch.arange(rnn_out.size(1), device=x.device)[None, :] >= lengths[:, None]
            rnn_out = rnn_out.masked_fill(mask.unsqueeze(-1), float('-inf'))

        pooled = torch.max(rnn_out, dim=1)[0]
        return self.fc(self.dropout(pooled)).squeeze(-1)