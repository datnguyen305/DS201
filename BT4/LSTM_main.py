# --- IMPORTS ---
from vocab import Vocab
from config.config import LSTM_config as Config
from models.LSTM import LSTM
from dataset import PhoMTDataset, collate_fn
from evaluate import Bleu, Rouge, Meteor

from torch.utils.data import DataLoader
from torch import nn 
import torch
import numpy as np 
from tqdm import tqdm 
import os 
import time

# --- CẤU HÌNH HỆ THỐNG ---
os.makedirs("checkpoints", exist_ok=True)
BEST_MODEL_PATH = "checkpoints/best_model.pt"
path = "/kaggle/input/phomt-dataset/dataset"

# --- KHỞI TẠO ---
print("Loading vocab ... ")
vocab = Vocab(path=path, src_language="vietnamese", tgt_language="english")

print("Creating model ... ")
config = Config()
model = LSTM(vocab, config).to(config.device)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx) 
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

print("Loading dataset ... ")
train_dataset = PhoMTDataset(path="/kaggle/input/phomt-dataset/dataset/train.json", vocab=vocab)
train_dataloader = DataLoader(
    dataset=train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    num_workers=config.num_workers, 
    collate_fn=collate_fn
)

dev_dataset = PhoMTDataset(path="/kaggle/input/phomt-dataset/dataset/dev.json", vocab=vocab)
dev_dataloader = DataLoader(
    dataset=dev_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=config.num_workers, 
    collate_fn=collate_fn
)

test_dataset = PhoMTDataset(path="/kaggle/input/phomt-dataset/dataset/test.json", vocab=vocab)   
test_dataloader = DataLoader(
    dataset=test_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=config.num_workers, 
    collate_fn=collate_fn
)

# --- HÀM ĐÁNH GIÁ (ĐÃ TỐI ƯU HÓA METRICS) ---
def evaluate(model, dataloader, vocab, device, metrics=None, src_language="vietnamese", tgt_language="english"):
    """
    metrics: list các string, ví dụ ['bleu'], ['bleu', 'rouge']. Nếu None hoặc rỗng thì chỉ tính Loss.
    """
    if metrics is None:
        metrics = []
        
    model.eval()
    total_loss = 0
    gens = {}  
    gts = {}   
    sample_index = 0 
    
    # Cờ để xem có cần chạy dự đoán (predict) hay không
    need_predictions = len(metrics) > 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch[src_language].to(device)
            tgt = batch[tgt_language].to(device)
            
            # 1. Tính LOSS (Luôn tính)
            loss = model(src, tgt) 
            total_loss += loss.item()
            
            # 2. Dự đoán (Chỉ chạy khi cần tính metrics)
            if need_predictions:
                prediction_tokens = model.predict(src) 
                
                prediction_sentences_list = vocab.decode_sentence(prediction_tokens, tgt_language)
                prediction_sentence = prediction_sentences_list[0]
                
                label_sentences_list = vocab.decode_sentence(tgt, tgt_language)
                label_sentence = label_sentences_list[0]

                # Lưu key dạng string cho thư viện eval
                id = str(sample_index)
                gens[id] = [prediction_sentence] 
                gts[id] = [label_sentence]     
                sample_index += 1
                
    avg_loss = total_loss / len(dataloader)
    metrics_scores = {}
    
    # 3. Tính toán Metrics (Chỉ tính những cái được yêu cầu)
    if need_predictions and len(gens) > 0:
        
        # --- TÍNH BLEU ---
        if 'bleu' in metrics:
            bleu_metric = Bleu()
            metrics_scores['BLEU'] = bleu_metric.compute_score(gts, gens) 
        
        # --- TÍNH ROUGE ---
        if 'rouge' in metrics:
            rouge_metric = Rouge()
            metrics_scores['ROUGE'] = rouge_metric.compute_score(gts, gens)
        
        # --- TÍNH METEOR ---
        if 'meteor' in metrics:
            try:
                meteor_metric = Meteor()
                metrics_scores['METEOR'] = meteor_metric.compute_score(gts, gens)
            except Exception as e:
                print(f"Skipping METEOR due to error: {e}")
                metrics_scores['METEOR'] = (0.0, 0.0)
            
    return avg_loss, metrics_scores


# --- VÒNG LẶP HUẤN LUYỆN CHÍNH ---

print("Starting training ... ")

best_bleu_score = -1.0 
patience_limit = 5  
patience_counter = 0

for epoch in range(config.num_epochs):
    start_time = time.time()
    model.train()
    total_loss = 0
    
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training")
    
    # 1. TRAINING LOOP
    for batch in train_bar:
        src = batch["vietnamese"].to(config.device)
        tgt = batch["english"].to(config.device)

        optimizer.zero_grad()
        loss = model(src, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_dataloader)
    
    # 2. EVALUATION TRÊN DEV (CHỈ TÍNH BLEU)
    # --- THAY ĐỔI TẠI ĐÂY: metrics=['bleu'] ---
    dev_loss, dev_metrics = evaluate(
        model, 
        dev_dataloader, 
        vocab, 
        config.device, 
        metrics=['bleu'] # Chỉ yêu cầu tính BLEU
    )
    
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    
    # 3. TRÍCH XUẤT BLEU-4
    current_bleu_4 = 0.0
    if 'BLEU' in dev_metrics:
        # BLEU trả về ([b1, b2, b3, b4], scores)
        score_list = dev_metrics['BLEU'][0]
        if isinstance(score_list, list) and len(score_list) >= 4:
            current_bleu_4 = score_list[3]

    print(f"\n--- Epoch {epoch+1:02d} Complete (Time: {epoch_mins}m {epoch_secs}s) ---")
    print(f"Training Loss: {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
    print(f"⭐️ Dev BLEU-4 Score: {current_bleu_4*100:.2f}%")

    # 4. EARLY STOPPING CHECK
    if current_bleu_4 > best_bleu_score:
        best_bleu_score = current_bleu_4
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  >>> SAVED: New best model found! (BLEU-4: {best_bleu_score*100:.2f}%)")
        patience_counter = 0 
    else:
        patience_counter += 1
        print(f"  >>> No improvement in BLEU. Patience: {patience_counter}/{patience_limit}")

    if patience_counter >= patience_limit:
        print(f"\n*** EARLY STOPPING TRIGGERED ***")
        break

# --- ĐÁNH GIÁ CUỐI CÙNG (TÍNH TẤT CẢ METRICS) ---

print("\nEvaluating on test set using the BEST saved model ... ")

try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print(f"Đã tải mô hình tốt nhất từ {BEST_MODEL_PATH}")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}. Sử dụng mô hình hiện tại.")

# Tính metrics
test_loss, test_metrics = evaluate(
    model, 
    test_dataloader, 
    vocab, 
    config.device, 
    metrics=['bleu', 'rouge', 'meteor']
)

print(f"\n--- FINAL TEST RESULTS ---")
print(f"Test Loss: {test_loss:.4f}")

# 1. IN ĐẦY ĐỦ BLEU-1, 2, 3, 4
if 'BLEU' in test_metrics:
    # test_metrics['BLEU'][0] chứa list [b1, b2, b3, b4]
    bleu_scores = test_metrics['BLEU'][0]
    
    if isinstance(bleu_scores, list) and len(bleu_scores) >= 4:
        print(f"BLEU-1: {bleu_scores[0]*100:.2f}%")
        print(f"BLEU-2: {bleu_scores[1]*100:.2f}%")
        print(f"BLEU-3: {bleu_scores[2]*100:.2f}%")
        print(f"BLEU-4: {bleu_scores[3]*100:.2f}%")
    else:
        print(f"BLEU Score: {bleu_scores}")

# 2. IN ROUGE
if 'ROUGE' in test_metrics:
    # ROUGE của bạn thường trả về ROUGE-L
    val = test_metrics['ROUGE'][0] if isinstance(test_metrics['ROUGE'], tuple) else test_metrics['ROUGE']
    print(f"ROUGE-L: {val*100:.2f}%")

# 3. IN METEOR
if 'METEOR' in test_metrics:
    val = test_metrics['METEOR'][0] if isinstance(test_metrics['METEOR'], tuple) else test_metrics['METEOR']
    print(f"METEOR: {val*100:.2f}%")