import torch
from datasets.av_dataset import AVDataset
from models.detector import get_model
import torchvision.transforms as T

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load test dataset
test_dir = "data/test/images"
test_ann = "data/test/annotations.json"
transform = T.Compose([T.ToTensor()])
test_dataset = AVDataset(test_dir, test_ann, transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load model
num_classes = 2
model = get_model(num_classes)
model.load_state_dict(torch.load("fasterrcnn_av.pth", map_location=device))
model.to(device)
model.eval()

results = {}
with torch.no_grad():
    for images, targets in test_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        for img_name, output in zip(targets, outputs):
            results[img_name['image_id']] = {
                "boxes": output['boxes'].cpu().numpy().tolist(),
                "scores": output['scores'].cpu().numpy().tolist()
            }

# Save prediction
import json
with open("submission.json", "w") as f:
    json.dump(results, f)
