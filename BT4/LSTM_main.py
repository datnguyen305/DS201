# --- IMPORTS ---
from vocab import Vocab
from config.config import LSTM_config as Config
from models.LSTM import LSTM
from dataset import PhoMTDataset, collate_fn

# Chỉ import Bleu, Rouge từ file cũ của bạn. BỎ Meteor cũ.
from custom_metric import Bleu, Rouge

# Import thư viện Hugging Face để tính METEOR không bị lỗi
import evaluate as hf_evaluate 
import nltk

from torch.utils.data import DataLoader
from torch import nn 
import torch
import numpy as np 
from tqdm import tqdm 
import os 
import time

# --- TẢI DỮ LIỆU NLTK (Cần thiết cho METEOR mới) ---
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

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

# Load METEOR metric của Hugging Face (Load 1 lần để dùng lại)
meteor_hf = hf_evaluate.load('meteor')

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

# --- HÀM ĐÁNH GIÁ (ĐÃ SỬA METEOR) ---
def evaluate_model(model, dataloader, vocab, device, metrics=None, src_language="vietnamese", tgt_language="english"):
    """
    Đổi tên hàm thành evaluate_model để tránh trùng tên với thư viện evaluate
    """
    if metrics is None:
        metrics = []
        
    model.eval()
    total_loss = 0
    gens = {}  
    gts = {}   
    sample_index = 0 
    
    need_predictions = len(metrics) > 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch[src_language].to(device)
            tgt = batch[tgt_language].to(device)
            
            # 1. Tính LOSS
            loss = model(src, tgt) 
            total_loss += loss.item()
            
            # 2. Dự đoán
            if need_predictions:
                prediction_tokens = model.predict(src) 
                
                prediction_sentences_list = vocab.decode_sentence(prediction_tokens, tgt_language)
                prediction_sentence = prediction_sentences_list[0]
                
                label_sentences_list = vocab.decode_sentence(tgt, tgt_language)
                label_sentence = label_sentences_list[0]

                id = str(sample_index)
                gens[id] = [prediction_sentence] 
                gts[id] = [label_sentence]     
                sample_index += 1
                
    avg_loss = total_loss / len(dataloader)
    metrics_scores = {}
    
    # 3. Tính toán Metrics
    if need_predictions and len(gens) > 0:
        
        # --- TÍNH BLEU (Dùng code cũ của bạn) ---
        if 'bleu' in metrics:
            bleu_metric = Bleu()
            metrics_scores['BLEU'] = bleu_metric.compute_score(gts, gens) 
        
        # --- TÍNH ROUGE (Dùng code cũ của bạn) ---
        if 'rouge' in metrics:
            rouge_metric = Rouge()
            metrics_scores['ROUGE'] = rouge_metric.compute_score(gts, gens)
        
        # --- TÍNH METEOR (Dùng thư viện mới Hugging Face) ---
        if 'meteor' in metrics:
            try:
                # Chuyển đổi dữ liệu từ dict sang list cho Hugging Face
                # gens[id] là list chứa 1 string -> lấy phần tử [0]
                predictions_list = [v[0] for v in gens.values()]
                references_list = [v[0] for v in gts.values()]
                
                # Tính điểm (trả về dictionary {'meteor': 0.xxxx})
                result = meteor_hf.compute(predictions=predictions_list, references=references_list)
                metrics_scores['METEOR'] = (result['meteor'], result['meteor']) # Lưu dạng tuple cho đồng bộ
            except Exception as e:
                print(f"Error computing METEOR: {e}")
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
    
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
    
    # 1. TRAINING LOOP
    for batch in train_bar:
        src = batch["vietnamese"].to(config.device)
        tgt = batch["english"].to(config.device)

        optimizer.zero_grad()
        loss = model(src, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Khuyên dùng: Clip grad để tránh bùng nổ gradient
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_dataloader)
    
    # 2. EVALUATION TRÊN DEV
    print(f"Evaluating Epoch {epoch+1}...")
    dev_loss, dev_metrics = evaluate_model(
        model, 
        dev_dataloader, 
        vocab, 
        config.device, 
        metrics=['bleu'] # Chỉ tính BLEU để tiết kiệm thời gian
    )
    
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    
    # 3. TRÍCH XUẤT BLEU-4
    current_bleu_4 = 0.0
    if 'BLEU' in dev_metrics:
        score_list = dev_metrics['BLEU'][0]
        if isinstance(score_list, list) and len(score_list) >= 4:
            current_bleu_4 = score_list[3]

    print(f"\n--- Epoch {epoch+1:02d} Complete (Time: {epoch_mins}m {epoch_secs}s) ---")
    print(f"Training Loss: {avg_train_loss:.4f} | Dev Loss: {dev_loss:.4f}")
    print(f"⭐️ Dev BLEU-4 Score: {current_bleu_4*100:.2f}%")

    # 4. EARLY STOPPING & PATIENCE CHECK (Phần bạn yêu cầu)
    if current_bleu_4 > best_bleu_score:
        best_bleu_score = current_bleu_4
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f" >>> SAVED: New best model found! (BLEU-4: {best_bleu_score*100:.2f}%)")
        patience_counter = 0 # Reset patience
    else:
        patience_counter += 1
        print(f" >>> No improvement. Patience: {patience_counter}/{patience_limit}")

    if patience_counter >= patience_limit:
        print(f"\n*** EARLY STOPPING TRIGGERED: Model did not improve for {patience_limit} epochs. ***")
        break

# --- ĐÁNH GIÁ CUỐI CÙNG ---

print("\nEvaluating on test set using the BEST saved model ... ")

try:
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print(f"Đã tải mô hình tốt nhất từ {BEST_MODEL_PATH}")
    else:
        print("Không tìm thấy file model checkpoint. Sử dụng model hiện tại.")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}. Sử dụng mô hình hiện tại.")

# Tính tất cả metrics cho test set
test_loss, test_metrics = evaluate_model(
    model, 
    test_dataloader, 
    vocab, 
    config.device, 
    metrics=['bleu', 'rouge', 'meteor'] 
)

print(f"\n--- FINAL TEST RESULTS ---")
print(f"Test Loss: {test_loss:.4f}")

# 1. IN ĐẦY ĐỦ BLEU
if 'BLEU' in test_metrics:
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
    val = test_metrics['ROUGE'][0] if isinstance(test_metrics['ROUGE'], tuple) else test_metrics['ROUGE']
    print(f"ROUGE-L: {val*100:.2f}%")

# 3. IN METEOR
if 'METEOR' in test_metrics:
    val = test_metrics['METEOR'][0] if isinstance(test_metrics['METEOR'], tuple) else test_metrics['METEOR']
    print(f"METEOR: {val*100:.2f}%")