import torch
from torch import nn
import torch.nn.functional as F

class GRUClassifier(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            num_layers: int,
            num_labels: int,
            padding_idx: int = 0
            ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
    
        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=num_labels
        )
    
    def forward(self, input_ids: torch.Tensor):
        embeddings = self.embedding(input_ids)
        _, features = self.gru(embeddings)

        features = features[-1] 
        
        logits = self.classifier(features)
        return logits