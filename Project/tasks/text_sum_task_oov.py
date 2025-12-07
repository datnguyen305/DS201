from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
import json
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from tasks.base_task import BaseTask
from data_utils import collate_fn_oov
import evaluation

@META_TASK.register()
class TextSumTaskOOV(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.training.score
        self.learning_rate = config.training.learning_rate
        self.patience = config.training.patience
        self.warmup = config.training.warmup

    def load_datasets(self, config):
        self.train_dataset = build_dataset(config.train, self.vocab)
        self.dev_dataset = build_dataset(config.dev, self.vocab)
        self.test_dataset = build_dataset(config.test, self.vocab)

    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn_oov
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn_oov
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn_oov
        )
    
    def get_vocab(self): 
        return self.vocab

    def train(self):
        self.model.train()
        train_losses = []
        running_loss = .0
        with tqdm(desc='Epoch %d - Training' % (self.epoch+1), unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                input_ids = items.input_ids.to(self.device)
                labels = items.shifted_right_label.to(self.device)
                extended_source_idx = items.extended_source_idx.to(self.device)
                extra_zeros = items.extra_zeros.to(self.device)
                _, loss = self.model(input_ids, labels, extended_source_idx, extra_zeros)
                
                # backward pass
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optim.step()
                current_loss = loss.item() # Lấy giá trị loss hiện tại
                running_loss += current_loss
                train_losses.append(current_loss) # <-- LƯU LOSS CỦA BATCH

                # update the training status
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()
        return train_losses

    def evaluate_metrics(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        gens = {}
        gts = {}
        all_oov_rates = []
        with tqdm(desc='Epoch %d - Evaluating' % (self.epoch+1), unit='it', total=len(dataloader)) as pbar:
            for items in dataloader:
                input_ids = items.input_ids.to(self.device)
                label = items.label.to(self.device)
                extended_source_idx = items.extended_source_idx.to(self.device)
                extra_zeros = items.extra_zeros.to(self.device)
                oov_list_batch = items.oov_list
                with torch.no_grad():
                    prediction_indices = self.model.predict(input_ids, extended_source_idx, extra_zeros)
                    batch_oov_rates = self.calculate_oov_rate(prediction_indices, oov_list_batch)
                    all_oov_rates.extend(batch_oov_rates) # Gom vào danh sách tổng
                    decoded_preds = self.decode_pgn_output(
                        prediction_indices, 
                        oov_list_batch
                    )

                    decoded_labels = self.decode_pgn_output(
                        label,
                        oov_list_batch
                    )

                    id = items.id[0]
                    gens[id] = decoded_preds[0]
                    gts[id] = decoded_labels[0]

                pbar.update()
        
        # Calculate metrics
        self.logger.info("Getting scores")
        scores = evaluation.compute_scores(gts, gens)
        if len(all_oov_rates) > 0:
            avg_oov_rate = sum(all_oov_rates) / len(all_oov_rates)
        else:
            avg_oov_rate = 0.0
        scores['avg_oov_rate'] = avg_oov_rate
    
        return scores, (gens, gts)

    def get_predictions(self):
        # 1. Tải checkpoint (Giữ nguyên)
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            self.logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        self.model.eval()
        
        # 2. Đánh giá trên tập TEST
        # Hàm evaluate_metrics sẽ tính ROUGE và thêm 'avg_oov_rate' vào scores
        scores, (gens, gts) = self.evaluate_metrics(self.test_dataloader) 
        
        results = {}
        with tqdm(desc='Epoch %d - Getting results' % (self.epoch+1), unit='it', total=len(gens)) as pbar:
            for id in gens:
                gen = gens[id]
                gt = gts[id]
                results[id] = {
                    "prediction": gen,
                    "target": gt
                }
                
                pbar.update()

        # 3. Lưu kết quả
        self.logger.info("Test scores %s", scores)
        
        # scores đã bao gồm 'avg_oov_rate'. File scores.json sẽ lưu tất cả.
        # Lưu file scores.json (bao gồm ROUGE và avg_oov_rate)
        json.dump(scores, open(os.path.join(self.checkpoint_path, "scores.json"), "w+"), ensure_ascii=False, indent=4) 
        
        # Lưu file predictions.json (kết quả dự đoán)
        json.dump(results, open(os.path.join(self.checkpoint_path, "predictions.json"), "w+"), ensure_ascii=False, indent=4)

    def decode_pgn_output(self, predicted_indices: torch.Tensor, oov_list_batch: list[list[str]]) -> list[str]:
        """
        Hàm mới để giải mã output của Pointer-Generator.
        Chuyển các index (bao gồm OOV) thành văn bản.
        """
        decoded_sentences = []
        vocab_size = self.vocab.vocab_size
        
        # Lặp qua từng câu trong batch
        # predicted_indices có shape (B, T)
        for i in range(predicted_indices.size(0)):
            indices = predicted_indices[i] # Lấy 1 câu (shape T)
            oov_list = oov_list_batch[i]   # Lấy list OOV của câu đó (ví dụ: ["gemini"])
            
            words = []
            for idx_tensor in indices:
                idx = idx_tensor.item()
                
                if idx < vocab_size:
                    # Đây là từ trong từ vựng cố định
                    word = self.vocab.itos[idx]
                    if word == self.vocab.eos_token:
                        break
                    if word != self.vocab.pad_token and word != self.vocab.bos_token:
                        words.append(word)
                else:
                    # Đây là từ OOV (con trỏ)
                    oov_local_index = idx - vocab_size
                    if oov_local_index < len(oov_list):
                        word = oov_list[oov_local_index]
                        words.append(word)
                    else:
                        # (Lỗi hiếm gặp) Con trỏ chỉ vào 1 index OOV không tồn tại
                        words.append(self.vocab.unk_token)
                        
            decoded_sentences.append(" ".join(words))
            
        return decoded_sentences
    
    def calculate_oov_rate(self, predicted_indices: torch.Tensor, oov_list_batch: list[list[str]]) -> list[float]:
        """
        Tính tỷ lệ từ OOV trong mỗi câu của batch.
        Trả về list[float] với mỗi phần tử là tỷ lệ OOV của một câu.
        """
        decoded_oov_rates = []
        vocab_size = self.vocab.vocab_size
        
        # predicted_indices có shape (B, T)
        for i in range(predicted_indices.size(0)):
            indices = predicted_indices[i] # Lấy 1 câu (shape T)
            oov_list = oov_list_batch[i] 
            
            total_words = 0
            oov_count = 0
            
            for idx_tensor in indices:
                idx = idx_tensor.item()
                
                # Bỏ qua các token đặc biệt (PAD, BOS, EOS)
                if idx == self.vocab.eos_idx or idx == self.vocab.pad_idx or idx == self.vocab.bos_idx:
                    continue
                
                total_words += 1
                
                if idx >= vocab_size:
                    # Đây là từ OOV (con trỏ)
                    oov_local_index = idx - vocab_size
                    
                    # Chỉ tính là OOV nếu nó thực sự trỏ tới một từ OOV hợp lệ
                    if oov_local_index < len(oov_list):
                        oov_count += 1
                    # Ngược lại, nếu trỏ ra ngoài list OOV, ta có thể coi là UNK/error
                    # Nhưng theo logic PGN, ta chỉ quan tâm đến các từ được sinh ra
            
            oov_rate = oov_count / total_words if total_words > 0 else 0.0
            decoded_oov_rates.append(oov_rate)
            
        return decoded_oov_rates