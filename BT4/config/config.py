import torch 

class LSTM_config:
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    bidirectional = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.001
    batch_size = 16
    num_epochs = 20
    