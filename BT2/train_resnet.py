from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from vinafood_dataset import VinaFood, collate_fn
from pretrained_resnet import PretrainedResnet
from tqdm import tqdm

# Config
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = (224, 224)

def evaluate(model, dataloader, device):
    model.eval()
    outputs = []
    trues = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=-1)

            outputs.extend(predictions.cpu().numpy())
            trues.extend(labels.cpu().numpy())
            
    return {
        "recall": recall_score(trues, outputs, average="macro", zero_division=0),
        "precision": precision_score(trues, outputs, average="macro", zero_division=0),
        "f1": f1_score(trues, outputs, average="macro", zero_division=0)
    }

def main():
    # Load datasets
    train_dataset = VinaFood(
        path="/content/drive/MyDrive/VinaFood21/train",
        image_size=image_size
    )
    
    val_dataset = VinaFood(
        path="/content/drive/MyDrive/VinaFood21/test",
        image_size=image_size,
        label2idx=train_dataset.label2idx  # Use same label mapping
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )

    # Initialize model
    model = PretrainedResnet(
        num_classes=len(train_dataset.label2idx),
        freeze_backbone=True  # Start with frozen backbone
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2, verbose=True
    )

    # Training loop
    best_f1 = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0
        
        # Training
        progress_bar = tqdm(train_loader, desc=f"Training")
        for batch in progress_bar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_loss:.4f}")

        # Validation
        metrics = evaluate(model, val_loader, device)
        print("Validation metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Learning rate scheduling
        scheduler.step(metrics['f1'])

        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'best_resnet.pth')
            print("Saved new best model!")

        # Unfreeze some layers after a few epochs
        if epoch == 5:  # After 5 epochs
            print("Unfreezing last few layers...")
            model.unfreeze_layers(num_layers=10)  # Unfreeze last 10 layers

if __name__ == "__main__":
    main()