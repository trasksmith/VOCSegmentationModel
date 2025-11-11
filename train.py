import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import fcn_resnet50
import torch.nn.functional as F
import datetime
from model import MobileNetV3_ASPP_Seg
import random
import torchvision.transforms.functional as T
import argparse

# Default hyperparameters
n_epochs = 25
batch_size = 16
learning_rate = 1e-4
num_classes = 21

def augment(image, mask,
                      flip_prob=0.5,
                      crop_size=(448, 448)):
    # Convert from PIL to tensor first
    image = T.to_tensor(image)
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)

    # Resize first for consistent shape
    image = T.resize(image, (512, 512))
    mask = T.resize(mask.unsqueeze(0).float(), (512, 512),
                    interpolation=T.InterpolationMode.NEAREST).squeeze(0).long()

    # Random horizontal flip
    if random.random() < flip_prob:
        image = T.hflip(image)
        mask = T.hflip(mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = T.crop(image, i, j, h, w)
    mask = T.crop(mask, i, j, h, w)

    # Random color jitter (applied only to image)
    if random.random() < 0.6:
        color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        image = color_jitter(image)


    # Normalize
    image = T.normalize(image, mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

    return image, mask


def voc_response_distillation_loss(student_model, teacher_model, images, masks, criterion, temperature=3.0, alpha=0.6,
                          ignore_index=255):

    student_model.train()
    teacher_model.eval()

    with torch.no_grad():
        teacher_outputs = teacher_model(images)['out']  # (B, 21, H, W)

    student_outputs = student_model(images)  # (B, 21, H, W)

    mask = masks != ignore_index  # (B, H, W)
    student_flat = student_outputs.permute(0, 2, 3, 1)[mask]  # (N_valid, 21)
    teacher_flat = teacher_outputs.permute(0, 2, 3, 1)[mask]  # (N_valid, 21)
    mask_flat = masks[mask]  # (N_valid,)

    # Hard label loss
    hard_loss = alpha * criterion(student_flat, mask_flat)

    # Soft label KD loss
    t_soft = F.softmax(teacher_flat / temperature, dim=1)
    s_log_soft = F.log_softmax(student_flat / temperature, dim=1)
    kd_loss = (1 - alpha) * F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (temperature ** 2)

    loss = hard_loss + kd_loss
    return loss, student_outputs

def voc_feature_distillation_loss(student_model, teacher_model, images, masks,
                                  criterion, alpha=0.6, beta=50.0, ignore_index=255):

    student_model.train()
    teacher_model.eval()

    with torch.no_grad():
        teacher_features = teacher_model.backbone(images)

    student_outputs, students_features = student_model(images, return_features=True)

    # ---------- Hard label loss ----------
    mask = masks != ignore_index
    student_flat = student_outputs.permute(0, 2, 3, 1)[mask]
    hard_loss = alpha * criterion(student_flat, masks[mask])

    # ---------- Feature KD loss ----------
    # (Pair student features with corresponding teacher layers)
    layer_pairs = [
        ('low', 'layer1'),
        ('mid', 'layer2'),
        ('high', 'layer3'),
        ('aspp', 'layer4')
    ]
    feat_loss = 0.0
    for s_key, t_key in layer_pairs:
        s_feat = students_features[s_key]
        t_feat = teacher_features[t_key]

        # Resize to match
        s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)

        # Match channels if necessary (temporary 1x1 projection)
        s_feat = student_model.feature_proj[s_key](s_feat)

        # Normalize before comparing
        s_norm = F.normalize(s_feat, p=2, dim=1)
        t_norm = F.normalize(t_feat, p=2, dim=1)
        cos_loss = 1 - F.cosine_similarity(s_norm, t_norm, dim=1).mean()

        feat_loss += cos_loss

    feat_loss = beta * feat_loss / len(layer_pairs)

    total_loss = hard_loss + feat_loss
    return total_loss, student_outputs

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set, do_augment =True):
        self.dataset = VOCSegmentation(root=root, year="2012", image_set=image_set)
        self.do_augment  = augment

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        if self.do_augment :
            image, mask = augment(image, mask)
        else:
            image = T.to_tensor(image)
            image = T.resize(image, (448, 448))
            image = T.normalize(image, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            mask = T.resize(mask, (448, 448), interpolation=T.InterpolationMode.NEAREST)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask

    def __len__(self):
        return len(self.dataset)

def train(student_model, teacher_model, train_loader, val_loader, optimizer, criterion, scheduler, train_miou, val_miou, device, save_file,
          loss_plot_file, miou_plot_file, mode):
    print('Starting training...')

    train_losses, val_losses = [], []
    train_mious, val_mious = [], []

    teacher_model.eval()

    for epoch in range(n_epochs):
        print(f"\n{datetime.datetime.now()}: Epoch [{epoch + 1}/{n_epochs}]")
        student_model.train()
        train_loss = 0.0
        train_miou.reset()

        # ---------------------- TRAIN LOOP ----------------------
        for imgs, masks in train_loader:

            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()

            if mode == "response":
                loss, outputs = voc_response_distillation_loss(student_model, teacher_model, imgs, masks, criterion)
            elif mode == "feature":
                loss, outputs = voc_feature_distillation_loss(student_model, teacher_model, imgs, masks, criterion)
            else:
                outputs = student_model(imgs)
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_miou.update(preds, masks)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_miou = train_miou.compute().item()

        # ---------------------- VALIDATION LOOP ----------------------
        student_model.eval()
        val_loss = 0.0
        val_miou.reset()

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device).long()

                outputs = student_model(imgs)
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
        torch.save(student_model.state_dict(), save_file)
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

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='state', type=str, default="None", help='Mode for KD')
    args = argParser.parse_args()

    if args.m not in ["response", "feature", "None"]:
        raise ValueError(f"Invalid KD mode: {args.m}. Must be 'response', 'feature', or 'None'.")
    else:
        mode = args.m


    save_file = 'weights_7.pth'
    loss_plot_file = 'loss_plot_7.png'
    miou_plot_file = 'miou_plot_7.png'

    print('Save File:', save_file)
    print('Loss Plot File:', loss_plot_file)
    print('Miou Plot File:', miou_plot_file)

    student_model = MobileNetV3_ASPP_Seg().to(device)

    teacher_model = fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1").to(device)
    teacher_model.eval()

    train_dataset = VOCDataset(
        root="./VOC2012_train_val/",
        image_set="train",
        do_augment=True
    )

    val_dataset = VOCDataset(
        root="./VOC2012_train_val/",
        image_set="val",
        do_augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    # Metrics
    train_miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)
    val_miou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)

    train(student_model, teacher_model, train_loader, val_loader, optimizer, criterion, scheduler, train_miou, val_miou, device, save_file,
          loss_plot_file, miou_plot_file, mode)

if __name__ == '__main__':
    main()
