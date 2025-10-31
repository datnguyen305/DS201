from torch.utils.data import DataLoader
from torch import nn 
import torch
import numpy as np 
from sklearn.metrics import precision_score, recall_score, f1_score
from vinafood_dataset import VinaFood, collate_fn
from ResNet_18 import ResNet18

device = "cpu"
image_size = (224, 224)

train_dataset = VinaFood(
    path=r"D:\NguyenTienDat_23520262\Nam_3\DL\BT2\VinaFood21\train",
    image_size=image_size
)

# test_dataset = VinaFood(
#     path=r"D:\NguyenTienDat_23520262\Nam_3\DL\BT2\VinaFood21\test",
#     image_size=image_size
# )

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=32,
#     collate_fn=collate_fn
# )
image_size = (3, ) + image_size   # (3, 224, 224)
model = ResNet18(num_labels=len(train_dataset.idx2label), image_size=image_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

EPOCHS = 10 
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1}")

    losses = []
    model.train() 
    for batch_idx, item in enumerate(train_loader):
        image = item["image"].to(device)
        label = item["label"].to(device)
        
        # Print shapes and values for first batch of each epoch
        if batch_idx == 0:
            print(f"\nImage shape: {image.shape}")
            print(f"Label shape: {label.shape}")
            print(f"Label values: {label.cpu().numpy()}")
        
        # Forward pass
        output = model(image)
        
        # Print output info for first batch
        if batch_idx == 0:
            print(f"Output shape: {output.shape}")
            print(f"Output sample: \n{output[0].cpu().detach().numpy()}\n")
        
        loss = loss_fn(output, label.long())
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {(np.array(losses).mean())}")
    metrics = evaluate(model, train_loader)
    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")
