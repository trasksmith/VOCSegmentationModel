import torch
import torch.nn as nn
import torchmetrics
from albumentations import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from model import MobileNetV3_ASPP_Seg
import random
import torchvision.transforms.functional as F
import albumentations as A

# Default hyperparameters
n_epochs = 20
batch_size = 16
learning_rate = 1e-4
num_classes = 21

'''
# --- Augmentation function ---
def geometric_augment(image, mask,
                      rotation_deg=15,
                      flip_prob=0.5,
                      crop_size=(224, 224)):

    # Random rotation
    angle = random.uniform(-rotation_deg, rotation_deg)
    image = F.rotate(image, angle)
    mask = F.rotate(mask.unsqueeze(0).float(), angle, interpolation=F.InterpolationMode.NEAREST).squeeze(0).long()

    # Random horizontal flip
    if random.random() < flip_prob:
        image = F.hflip(image)
        mask = F.hflip(mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = F.crop(image, i, j, h, w)
    mask = F.crop(mask, i, j, h, w)

    image = F.resize(image, (256, 256))
    mask = F.resize(mask.unsqueeze(0).float(), (256, 256), interpolation=F.InterpolationMode.NEAREST).squeeze(0).long()

    # Normalize image
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return image, mask
'''


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set, transform=None):
        self.dataset = VOCSegmentation(root=root, year="2012", image_set=image_set)
        self.transform = transform

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = np.array(image)
        mask = np.array(mask)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        return image, mask

    def __len__(self):
        return len(self.dataset)


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, train_miou, val_miou, device, save_file,
          loss_plot_file, miou_plot_file):
    print('Starting training...')

    train_losses, val_losses = [], []
    train_mious, val_mious = [], []

    for epoch in range(n_epochs):
        print(f"\nEpoch [{epoch + 1}/{n_epochs}]")
        model.train()
        train_loss = 0.0
        train_miou.reset()

        # ---------------------- TRAIN LOOP ----------------------
        for imgs, masks in train_loader:
            '''
            batch_imgs = []
            batch_masks = []

            for img, mask in zip(imgs, masks):
                img_pil = F.to_pil_image(img)
                mask_pil = Image.fromarray(mask.numpy().astype(np.uint8))

                img_aug, mask_aug = geometric_augment(img_pil, mask_pil)

                batch_imgs.append(img_aug)
                batch_masks.append(mask_aug)
            '''

            imgs, masks = imgs.to(device), masks.to(device)

            # imgs = torch.stack(imgs).to(device)
            # masks = torch.stack(masks).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_miou.update(preds, masks)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_miou = train_miou.compute().item()

        # ---------------------- VALIDATION LOOP ----------------------
        model.eval()
        val_loss = 0.0
        val_miou.reset()

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device).long()
                outputs = model(imgs)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_miou.update(preds, masks)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_miou = val_miou.compute().item()

        # Scheduler step
        scheduler.step(epoch_val_loss)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_mious.append(epoch_train_miou)
        val_mious.append(epoch_val_miou)

        print(
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Train mIoU: {epoch_train_miou:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val mIoU: {epoch_val_miou:.4f}"
        )

    # Save weights and loss plot
    if save_file != None:
        torch.save(model.state_dict(), save_file)
    if loss_plot_file != None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        plt.savefig(loss_plot_file)
        plt.close()

    if miou_plot_file != None:
        plt.figure(figsize=(10, 6))
        plt.plot(train_mious, label='Train mIoU')
        plt.plot(val_mious, label='Val mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.legend()
        plt.title('Segmentation Accuracy')
        plt.savefig(miou_plot_file)
        plt.close()

    print('Training complete! Model saved as', save_file)
    print('Loss plot saved as', loss_plot_file)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    save_file = 'weights.pth'
    loss_plot_file = 'loss_plot.png'
    miou_plot_file = 'miou_plot.png'

    print('Save File:', save_file)
    print('Loss Plot File:', loss_plot_file)
    print('Miou Plot File:', miou_plot_file)

    model = MobileNetV3_ASPP_Seg().to(device)
    '''
    val_image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.long))
    ])

    train_image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_mask_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.as_tensor(np.array(x), dtype=torch.long))
    ])
    '''
    train_transform = A.Compose([
        A.Rotate(limit=15, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.Resize(256, 256, interpolation=1),  # bilinear for image
        A.RandomCrop(224, 224, p=1.0),
        A.Resize(256, 256, interpolation=1),  # bilinear for image
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})

    val_transform = A.Compose([
        A.Resize(256, 256, interpolation=1),  # Resize to model input size
        A.Normalize(mean=(0.485, 0.456, 0.406),  # Standard ImageNet normalization
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()  # Converts image to PyTorch tensor (C, H, W)
    ], additional_targets={"mask": "mask"})

    val_dataset = VOCDataset(
        root="./VOC2012_train_val/",
        image_set="val",
        transform=val_transform
    )

    train_dataset = VOCDataset(
        root="./VOC2012_train_val/",
        image_set="train",
        transform=train_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    # Metrics
    # train_miou = MeanIoU(num_classes=num_classes).to(device)
    # val_miou = MeanIoU(num_classes=num_classes).to(device)
    train_miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)
    val_miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)

    train(model, train_loader, val_loader, optimizer, criterion, scheduler, train_miou, val_miou, device, save_file,
          loss_plot_file, miou_plot_file)


if __name__ == '__main__':
    main()
