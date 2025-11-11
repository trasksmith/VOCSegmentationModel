import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from torchmetrics.segmentation import MeanIoU
import os
from PIL import Image
import numpy as np
from torchvision.datasets import VOCSegmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 21  # 20 classes + background for VOC2012

def evaluate(loader, model):
    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            print(f"Loading batch {idx}...")  # <- DEBUG PRINT
            images = images.to(DEVICE)
            targets = targets.squeeze(1).to(DEVICE)

            print(f"Running model on batch {idx}...")  # <- DEBUG PRINT
            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1)

            print(f"Updating mIoU for batch {idx}...")  # <- DEBUG PRINT
            miou.update(preds, targets)

    mean_iou = miou.compute()
    print(f"Mean IoU: {mean_iou.item():.4f}")
    miou.reset()

if __name__ == "__main__":
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.long))
    ])

    val_dataset = VOCSegmentation(
        root="./VOC2012_train_val/",
        year="2012",
        image_set="val",
        download=False,
        transform=image_transform,
        target_transform=mask_transform
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")
    model = model.to(DEVICE)
    model.eval()

    miou = MeanIoU(num_classes=NUM_CLASSES).to(DEVICE)
    evaluate(val_loader, model)

