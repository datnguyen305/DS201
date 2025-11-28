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

path = "/kaggle/input/phomt-dataset/dataset"

print("Loading vocab ... ")
vocab = Vocab(path=path, src_language="vietnamese", tgt_language="english")

print("Creating model ... ")
config = Config()
model = LSTM(vocab, config).to(config.device)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
print("Loading dataset ... ")
train_dataset = PhoMTDataset(
    path="/kaggle/input/phomt-dataset/dataset/train.json",
    vocab=vocab
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

dev_dataset = PhoMTDataset(
    path="/kaggle/input/phomt-dataset/dataset/dev.json",
    vocab=vocab
)
dev_dataloader = DataLoader(
    dataset=dev_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

test_dataset = PhoMTDataset(
    path="/kaggle/input/phomt-dataset/dataset/test.json",
    vocab=vocab
)   
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

def evaluate(model, dataloader, vocab, device, deeper_evaluate=False, src_language="vietnamese", tgt_language="english"):
    """
    Đánh giá mô hình trên tập dữ liệu phát triển hoặc kiểm tra.

    Args:
        model (nn.Module): Mô hình Seq2Seq (LSTM).
        dataloader (DataLoader): DataLoader cho tập Dev hoặc Test (batch_size=1).
        vocab (Vocab): Đối tượng từ vựng để giải mã.
        device (torch.device): Thiết bị (CPU/GPU).
        deeper_evaluate (bool): True nếu muốn tính BLEU, ROUGE, METEOR.

    Returns:
        tuple: (average_loss, metrics_scores)
    """
    # 1. Đặt mô hình ở chế độ đánh giá
    model.eval()
    total_loss = 0
    
    # Khởi tạo từ điển để lưu trữ câu dự đoán và câu tham chiếu
    gens = {}  # Generations (câu dự đoán)
    gts = {}   # Ground Truths (câu tham chiếu)

    sample_index = 0 # Chỉ mục bắt đầu từ 0
    
    # 2. Bắt đầu quá trình đánh giá
    with torch.no_grad():
        for batch in dataloader:
            # Dữ liệu được chuyển lên device (config.device)
            src = batch[src_language].to(device)
            tgt = batch[tgt_language].to(device)
            
            # --- TÍNH LOSS ---
            # Gọi forward pass của mô hình. Dựa trên mô tả của bạn, hàm này trả về loss trực tiếp.
            loss = model(src, tgt) 
            total_loss += loss.item()
            
            
            # --- THỰC HIỆN DỰ ĐOÁN (Inference) cho Deeper Evaluation ---
            if deeper_evaluate:
                # 3. Lấy token dự đoán
                # Giả định model.predict(src) trả về tensor chứa các token dự đoán (batch_size=1)
                prediction_tokens = model.predict(src) 
                
                # 4. Giải mã và lưu trữ
                
                # prediction_tokens: tensor (1, seq_len). Ta lấy .tolist() của tensor đầu tiên.
                prediction_sentence = vocab.decode_sentence(prediction_tokens[0].tolist())
                
                # label_tokens: tensor tgt[0] chứa các ID target (bao gồm <sos> và <eos>)
                label_tokens = tgt[0].tolist() 
                label_sentence = vocab.decode_sentence(label_tokens)

                # Vì batch_size=1, ta chỉ lấy phần tử đầu tiên
                id = sample_index # Giả định batch có key "id"
                gens[id] = [prediction_sentence] # Danh sách các câu dự đoán
                gts[id] = [label_sentence]      # Danh sách các câu tham chiếu (reference)
                sample_index += 1
                
    # 5. Tính Loss trung bình
    avg_loss = total_loss / len(dataloader)
    
    # 6. --- TÍNH TOÁN METRICS (BLEU, ROUGE, METEOR) ---
    metrics_scores = {}
    if deeper_evaluate:
        # Tạo list câu dự đoán và tham chiếu theo thứ tự ID
        ids = sorted(gens.keys())
        # Tạo list predictions (gens[i] là list ['câu dự đoán'])
        predictions = [gens[i][0] for i in ids] 
        # Tạo list references (gts[i] là list ['câu tham chiếu'])
        references = [[gts[i][0]] for i in ids] # references cần là list of list

        # Tính điểm BLEU
        bleu_metric = Bleu()
        # compute_score trả về tuple: (bleu_scores, cumulative_scores)
        metrics_scores['BLEU'] = bleu_metric.compute_score(predictions, references) 
        
        # Tính điểm ROUGE
        rouge_metric = Rouge()
        # compute_score trả về tuple: (rouge_l_score)
        metrics_scores['ROUGE'] = rouge_metric.compute_score(predictions, references)
        
        # Tính điểm METEOR
        meteor_metric = Meteor()
        # compute_score trả về tuple: (meteor_score)
        metrics_scores['METEOR'] = meteor_metric.compute_score(predictions, references)
            
    return avg_loss, metrics_scores


print("Starting training ... ")
for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    
    # ⭐️ THÊM THANH TIẾN TRÌNH TẠI ĐÂY ⭐️
    # Wrap train_dataloader bằng tqdm()
    # Thêm mô tả (desc) để biết đây là epoch nào
    # Thêm thanh tiến trình (bar) để hiển thị loss hiện tại
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training")
    
    for batch in train_bar:
        src = batch["vietnamese"].to(config.device)
        tgt = batch["english"].to(config.device)

        optimizer.zero_grad()
        loss = model(src, tgt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Cập nhật thông tin trên thanh tiến trình với loss hiện tại
        current_loss = loss.item()
        train_bar.set_postfix(loss=f"{current_loss:.4f}")
    
    # Tính toán và in ra average loss sau khi hoàn thành epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch}, Training Loss: {avg_loss:.4f}")
    
    # Đánh giá trên tập Dev
    dev_loss, _ = evaluate(model, dev_dataloader, vocab, config.device, deeper_evaluate=False)
    print(f"Epoch {epoch}, Dev Loss: {dev_loss:.4f}")

print("Evaluating on test set ... ")
test_loss, test_metrics = evaluate(model, test_dataloader, vocab, config.device, deeper_evaluate=True)
print(f"Test Loss: {test_loss:.4f}")
for metric_name, score in test_metrics.items():
    print(f"{metric_name} Score: {score:.4f}")



