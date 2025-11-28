import torch 
import torch.nn as nn
from vocab import Vocab

class Encoder(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.config = config 
        self.embedding = nn.Embedding(vocab.src_vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(config.embedding_dim,
                            config.hidden_dim,
                            num_layers=config.num_layers,
                            dropout=config.dropout,
                            bidirectional=config.bidirectional,
                            batch_first=True)
        
    def forward(self, input):
        embedded = self.embedding(input)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, vocab, config): 
        super().__init__()
        self.config = config 
        self.embedding = nn.Embedding(vocab.tgt_vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim*2 if config.bidirectional else config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first=True
        )
    def forward(self, encoder_outputs, states, target):
        batch_size = encoder_outputs.size(0)
        target_len = target.size(1)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(vocab.Vocab.bos_idx).to(self.config.device)
        decoder_outputs = []
        for i in range(target_len):
            decoder_output, states = self.forward_step(decoder_input, states)
            decoder_outputs.append(decoder_output)
            # Here comes the teacher forcing
            decoder_input = target[:, i].unsqueeze(1)
        return decoder_outputs


    def forward_step(self, input, states):
        embedded = self.embedding(input)
        outputs, states = self.lstm(embedded, states)
        return outputs, states
    
class LSTM(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.encoder = Encoder(vocab, config)
        self.decoder = Decoder(vocab, config)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, src, tgt):
        encoder_outputs, states = self.encoder(src)
        outs, _ = self.decoder(encoder_outputs, states, tgt)
        loss = self.loss(outs.reshape(-1, self.vocab.tgt_vocab_size), tgt.reshape(-1)) # loss input: (N, C), target: (N)
        return loss
    
    def predict(self, src): 
        encoder_outputs, states = self.encoder(src)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(vocab.Vocab.bos_idx).to(self.config.device)
        predictions = []
        while True:
            decoder_output, states = self.decoder.forward_step(decoder_input, states)
            pred_token = decoder_output.argmax(-1)  # (batch_size, 1)
            predictions.append(pred_token)
            decoder_input = pred_token
            if pred_token.eq(vocab.Vocab.eos_idx).all():
                break
        predictions = torch.cat(predictions, dim=1)  # (batch_size, max_tgt_len)
        return predictions