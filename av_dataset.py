import torch
from torch.utils.data import Dataset
import os
import cv2
import json

class AVDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        self.imgs = list(self.annotations.keys())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = torch.tensor(self.annotations[img_name]['boxes'], dtype=torch.float32)
        labels = torch.tensor(self.annotations[img_name]['labels'], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target
