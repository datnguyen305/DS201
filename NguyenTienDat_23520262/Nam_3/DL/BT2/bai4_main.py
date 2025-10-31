from torch.utils.data import DataLoader
from torch import nn 
import torch
import numpy as np 
from sklearn.metrics import precision_score, recall_score, f1_score
from vinafood_dataset import VinaFood, collate_fn
from pretrained_resnet import PretrainedResnet
from tqdm import tqdm

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 5e-5  
EPOCHS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = (224, 224)

# Load datasets
train_dataset = VinaFood(
    path=r"D:\NguyenTienDat_23520262\Nam_3\DL\BT2\VinaFood21\train",
    image_size=image_size
)

# val_dataset = VinaFood(
#     path=r"D:\NguyenTienDat_23520262\Nam_3\DL\BT2\VinaFood21\test",
#     image_size=image_size,
#     label2idx=train_dataset.label2idx  # Use same label mapping
# )

# Create data loaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# val_loader = DataLoader(
#     dataset=val_dataset,
#     batch_size=BATCH_SIZE,
#     collate_fn=collate_fn
# )

# Initialize model
model = PretrainedResnet(
    num_classes=len(train_dataset.label2idx),
    freeze_backbone=True  # Start with frozen backbone
).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=2, verbose=True
)

def evaluate(model, dataloader):
    model.eval() 
    outputs = []
    trues = []
    with torch.no_grad():
        for item in dataloader:
            image = item["image"].to(device) 
            label = item["label"].to(device) 
            output = model(image)   
            predictions = torch.argmax(output, dim=-1)  

            outputs.extend(predictions.cpu().numpy())
            trues.extend(label.cpu().numpy())
    
    # Print unique values for debugging
    print(f"Unique predictions: {np.unique(outputs)}")
    print(f"Unique true labels: {np.unique(trues)}")
    
    try:
        return {
            "recall": recall_score(trues, outputs, average="macro", zero_division=0),
            "precision": precision_score(trues, outputs, average="macro", zero_division=0),
            "f1": f1_score(trues, outputs, average="macro", zero_division=0),
        }
    except Exception as e:
        print(f"Error in metrics calculation: {e}")
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0
        }

# Training loop
best_f1 = 0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0
    
    # Training
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_loss:.4f}")

    # Validation
    metrics = evaluate(model, train_loader)
    print("Validation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # # Learning rate scheduling
    # scheduler.step(metrics['f1'])

    # # Save best model
    # if metrics['f1'] > best_f1:
    #     best_f1 = metrics['f1']
    #     torch.save(model.state_dict(), 'best_resnet.pth')
    #     print("Saved new best model!")

    # Unfreeze some layers after a few epochs
    if epoch == 5:  # After 5 epochs
        print("\nUnfreezing last few layers...")
        model.unfreeze_layers(num_layers=10)  # Unfreeze last 10 layers
