import torch
from torch.utils.data import DataLoader
from datasets.av_dataset import AVDataset
from models.detector import get_model
import torchvision.transforms as T

# Dataset paths
train_dir = "data/train/images"
train_ann = "data/train/annotations.json"

# Dataset and DataLoader
transform = T.Compose([T.ToTensor()])
train_dataset = AVDataset(train_dir, train_ann, transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model
num_classes = 2  # 背景 + 主動脈瓣
model = get_model(num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {losses.item():.4f}")

# Save model
torch.save(model.state_dict(), "fasterrcnn_av.pth")
