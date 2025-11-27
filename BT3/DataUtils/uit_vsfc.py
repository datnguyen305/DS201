import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import json
import os
import string

def collate_fn(items: dict) -> torch.Tensor:
    input_ids = [item["input_ids"] for item in items]
    max_len = max(len(ids) for ids in input_ids)
    input_ids = [
        F.pad(
            input,
            pad = (0, max(0, max_len - input.shape[0])),
            mode = "constant",
            value = 0
        ).unsqueeze(0) for input in input_ids]
    
    label_ids = [item["label_ids"].unsqueeze(0) for item in items]

    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "label_ids": torch.cat(label_ids, dim=0)
    }

class Vocab:
    def __init__(self, path: list[str]):
        all_words = set()
        labels = set()
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                sentence: str = item["sentence"]
                sentence = self._preprocess_sentence(sentence)
                all_words.update(sentence.split())
                labels.add(item["topic"])
        
        self.bos = "<s>"
        self.pad = "<p>"

        self.word2idx = {
            word: idx for idx, word in enumerate(all_words, start=2)
        }
        self.word2idx[self.bos] = 1
        self.word2idx[self.pad] = 0
        self.idx2word = {
            idx: word for word, idx in self.word2idx.items()
        }

        self.label2idx = {
            label: idx for idx, label in enumerate(labels)
        }
        self.idx2label = {
            idx: label for label, idx in self.label2idx.items()
        }

    @property
    def n_labels(self) -> int:
        return len(self.label2idx)
    
    @property
    def len(self) -> int:
        return len(self.word2idx)
    
    def _preprocess_sentence(self, sentence: str) -> str:
        translator = str.maketrans('', '', string.punctuation)
        sentence = sentence.lower()
        sentence = sentence.translate(translator)
        return sentence

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        sentence = self._preprocess_sentence(sentence)
        words = sentence.split()
        words = [self.bos] + words
        word_ids =[self.word2idx[word] for word in words]

        return torch.tensor(word_ids, dtype=torch.long)

    def encode_label(self, label: str) -> torch.Tensor:
        label_idx = self.label2idx[label]
        return torch.tensor(label_idx)
    
    def decode_label(self, label_ids: torch.Tensor) -> str:
        label_ids = label_ids.tolist()
        labels = [self.idx2label[label_id] for label_id in label_ids]
        return labels
        
class UIT_VSFC(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict:
        item = self.data[index]

        sentence: str = item["sentence"]
        label: str = item["topic"]
        
        input_ids = self.vocab.encode_sentence(sentence)
        label_ids = self.vocab.encode_label(label) 

        return {
            "input_ids": input_ids,
            "label_ids": label_ids
        }
