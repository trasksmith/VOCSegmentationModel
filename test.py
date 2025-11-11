import torch
from torchmetrics import JaccardIndex
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import numpy as np
from model import MobileNetV3_ASPP_Seg
from torchmetrics.segmentation import MeanIoU
import matplotlib.pyplot as plt
from PIL import Image
from train import VOCDataset

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    model = MobileNetV3_ASPP_Seg()
    model.load_state_dict(torch.load('weights_3.pth', map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    val_dataset = VOCDataset(
        root="./VOC2012_train_val/",
        image_set="val",
        do_augment=False
    )

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_miou = JaccardIndex(task="multiclass", num_classes=21, ignore_index=255).to(device)

    with torch.no_grad():
        test_miou.reset()
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device).long()
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            test_miou.update(preds, masks)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Mean IoU on test set: {test_miou.compute().item():.4f}")
    print(f"Total parameters: {total_params:,}")

    idx = 0
    while True:
        idx = input("Enter index (or -1 to exit) to see an image > ")
        idx = int(idx)
        if idx < 0 or idx >= len(val_dataset):
            break

        img, mask = val_dataset[idx]
        img_input = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_input)
            pred_mask = torch.argmax(pred, dim=1).squeeze().cpu()

        # Define mean and std to denormalize
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        # Denormalize for display
        img_disp = img.cpu().permute(1, 2, 0) * std + mean
        img_disp = img_disp.numpy()
        img_disp = np.clip(img_disp, 0, 1)  # Clip values between 0 and 1 for imshow
        gt_mask = mask.cpu()

        cmap = plt.get_cmap('tab20')  # discrete colormap for up to 20 classes

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_disp)
        axes[0].set_title("Input Image")
        axes[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=20)
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=20)
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()