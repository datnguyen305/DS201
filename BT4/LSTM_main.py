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
# loss_fn chỉ dùng bên trong class LSTM nếu cần, ở đây ta khai báo để tường minh nhưng model tự xử lý
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx) 
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

print("Loading dataset ... ")
train_dataset = PhoMTDataset(path="/kaggle/input/phomt-dataset/dataset/train.json", vocab=vocab)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_fn)

dev_dataset = PhoMTDataset(path="/kaggle/input/phomt-dataset/dataset/dev.json", vocab=vocab)
dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

test_dataset = PhoMTDataset(path="/kaggle/input/phomt-dataset/dataset/test.json", vocab=vocab)   
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

# --- HÀM ĐÁNH GIÁ (ĐÃ SỬA: Loại bỏ tham số loss_fn thừa) ---
def evaluate(model, dataloader, vocab, device, deeper_evaluate=False, src_language="vietnamese", tgt_language="english"):
    model.eval()
    total_loss = 0
    gens = {}  
    gts = {}   
    sample_index = 0 
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch[src_language].to(device)
            tgt = batch[tgt_language].to(device)
            
            # Tính LOSS
            loss = model(src, tgt) 
            total_loss += loss.item()
            
            if deeper_evaluate:
                prediction_tokens = model.predict(src) 
                
                # --- SỬA LỖI TẠI ĐÂY ---
                # Thêm tham số tgt_language vào decode_sentence
                
                # Giải mã câu dự đoán (Prediction) - Dùng ngôn ngữ đích (English)
                prediction_sentence = vocab.decode_sentence(prediction_tokens[0], tgt_language)
                
                # Giải mã câu nhãn (Label) - Dùng ngôn ngữ đích (English)
                label_tokens = tgt[0].tolist() 
                label_sentence = vocab.decode_sentence(label_tokens, tgt_language)

                id = sample_index
                gens[id] = [prediction_sentence] 
                gts[id] = [label_sentence]     
                sample_index += 1
                
    avg_loss = total_loss / len(dataloader)
    metrics_scores = {}
    
    if deeper_evaluate and len(gens) > 0:
        ids = sorted(gens.keys())
        predictions = [gens[i][0] for i in ids] 
        references = [[gts[i][0]] for i in ids] 

        # Tính toán Metrics
        bleu_metric = Bleu()
        metrics_scores['BLEU'] = bleu_metric.compute_score(predictions, references) 
        
        rouge_metric = Rouge()
        metrics_scores['ROUGE'] = rouge_metric.compute_score(predictions, references)
        
        meteor_metric = Meteor()
        metrics_scores['METEOR'] = meteor_metric.compute_score(predictions, references)
            
    return avg_loss, metrics_scores


# --- VÒNG LẶP HUẤN LUYỆN CHÍNH ---

print("Starting training ... ")

# ⭐️ CẤU HÌNH PATIENCE (EARLY STOPPING) DỰA TRÊN BLEU ⭐️
# BLEU càng cao càng tốt, nên khởi tạo là số rất nhỏ
best_bleu_score = -1.0 
patience_limit = 5  # Dừng nếu sau 5 epoch BLEU không tăng
patience_counter = 0

for epoch in range(config.num_epochs):
    start_time = time.time()
    model.train()
    total_loss = 0
    
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training")
    
    # 1. TRAINING
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
    
    # 2. EVALUATION (Bắt buộc bật deeper_evaluate=True để lấy BLEU)
    # Lưu ý: Không truyền loss_fn vào đây nữa
    dev_loss, dev_metrics = evaluate(
        model, 
        dev_dataloader, 
        vocab, 
        config.device, 
        deeper_evaluate=True 
    )
    
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    
    # 3. TRÍCH XUẤT BLEU-4
    current_bleu_4 = 0.0
    if 'BLEU' in dev_metrics:
        # metrics['BLEU'][0] là mảng [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
        # Ta lấy phần tử index 3 là BLEU-4
        current_bleu_4 = dev_metrics['BLEU'][0][3]

    print(f"\n--- Epoch {epoch+1:02d} Complete (Time: {epoch_mins}m {epoch_secs}s) ---")
    print(f"Training Loss: {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
    print(f"⭐️ Dev BLEU-4 Score: {current_bleu_4*100:.2f}%")

    # 4. LOGIC PATIENCE (Dựa trên BLEU-4)
    if current_bleu_4 > best_bleu_score:
        best_bleu_score = current_bleu_4
        # Lưu mô hình tốt nhất
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  >>> SAVED: New best model found! (BLEU-4: {best_bleu_score*100:.2f}%)")
        patience_counter = 0 # Reset bộ đếm
    else:
        patience_counter += 1
        print(f"  >>> No improvement in BLEU. Patience: {patience_counter}/{patience_limit}")

    # 5. KIỂM TRA DỪNG SỚM
    if patience_counter >= patience_limit:
        print(f"\n*** EARLY STOPPING TRIGGERED ***")
        print(f"Mô hình không cải thiện BLEU sau {patience_limit} epoch liên tiếp.")
        break

# ----------------------------------------------------------------------
# --- ĐÁNH GIÁ CUỐI CÙNG ---

print("\nEvaluating on test set using the BEST saved model ... ")

try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print(f"Đã tải mô hình tốt nhất từ {BEST_MODEL_PATH}")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}. Sử dụng mô hình hiện tại.")

test_loss, test_metrics = evaluate(model, test_dataloader, vocab, config.device, deeper_evaluate=True)

print(f"\n--- FINAL TEST RESULTS ---")
print(f"Test Loss: {test_loss:.4f}")
if 'BLEU' in test_metrics:
    print(f"BLEU-4 Score: {test_metrics['BLEU'][0][3]*100:.2f}%")
if 'ROUGE' in test_metrics:
    # ROUGE thường trả về 1 số float hoặc tuple, xử lý tùy library
    val = test_metrics['ROUGE'][0] if isinstance(test_metrics['ROUGE'], tuple) else test_metrics['ROUGE']
    print(f"ROUGE-L Score: {val*100:.2f}%")
if 'METEOR' in test_metrics:
    val = test_metrics['METEOR'][0] if isinstance(test_metrics['METEOR'], tuple) else test_metrics['METEOR']
    print(f"METEOR Score: {val*100:.2f}%")