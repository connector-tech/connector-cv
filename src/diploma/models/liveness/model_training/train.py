import yaml
import torch

import pandas as pd
import torch.nn as nn
import albumentations as A
import torchvision.models as models

from tqdm import tqdm
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader

# from sklearn.model_selection import train_test_split

from diploma.models.liveness.model_training.dataset import PersonDataset
from diploma.models.liveness.model_training.utils import (
    calculate_metrics,
    calculate_roc_auc,
    calculate_average_precision,
    calculate_confusion_matrix,
)
from torch.utils.tensorboard import SummaryWriter


def focal_loss(outputs, labels, alpha=0.25, gamma=2):
    bce_loss = F.binary_cross_entropy_with_logits(outputs, labels)
    weights = torch.pow(1 - torch.abs(labels - 0.5), gamma)
    focal_loss = alpha * weights * bce_loss
    return focal_loss.mean()


def train(config_path, log_dir):
    name = input()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = int(cfg["Training"]["epochs"])
    learning_rate = float(cfg["Training"]["learning_rate"])
    device = torch.device(cfg["device"])

    train_data_path = cfg["Data"]["train_data_path"]
    val_data_path = cfg["Data"]["val_data_path"]

    IMSIZE = cfg["imsize"]
    MEAN = cfg["mean"]
    STD = cfg["std"]
    BATCH_SIZE = cfg["batch_size"]
    SHUFFLE = cfg["shuffle"]
    NUM_WORKERS = cfg["num_workers"]
    WEIGHTS = cfg["weights"]
    enable = cfg["autocast_enabled"]

    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)

    # df = pd.read_csv(val_data_path)
    # train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # vozmozhno nado budet eshe dobavit hz posmotrim
    transform = A.Compose(
        [
            A.Resize(IMSIZE, IMSIZE, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.4),
            A.Normalize(mean=MEAN, std=STD, p=1),
        ]
    )

    transform_val = A.Compose(
        [
            A.Resize(IMSIZE, IMSIZE, p=1),
            A.Normalize(mean=MEAN, std=STD, p=1),
        ]
    )

    train_data = PersonDataset(train_df, transform)
    val_data = PersonDataset(val_df, transform_val)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # nado budet posmetret drugie variki elsi ne budet hvatat
    # model = models.mobilenet_v2(weights=WEIGHTS)
    # model = models.efficientnet_b0(weights=WEIGHTS)
    model = models.efficientnet_b1(weights=WEIGHTS)
    # model = model.half()  # - po idee naxyi ne nado

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.to(device)

    criterion = focal_loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2
    )

    writer = SummaryWriter(log_dir)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        model.train()

        for i, (images, labels) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            scaler = amp.GradScaler()

            with amp.autocast(enabled=enable):
                images = images.to(device).float()
                labels = labels.to(device).float()

                outputs = model(images)
                outputs = torch.squeeze(outputs, dim=1)

                loss = criterion(outputs, labels)

                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.step(optimizer)
                scaler.update()

            if (i + 1) % 100 == 0:
                print(
                    f"Training: Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)

        model.eval()
        # vozmozhno dlia val-a tozhe nado autocast, nado proverit budet hotia poh mozhno i bez
        with torch.no_grad():
            val_loss = 0
            val_accuracy = 0
            val_precision = 0
            val_recall = 0
            val_f1_score = 0
            val_cm = torch.zeros((2, 2), dtype=torch.int64)
            val_roc_auc = 0
            val_average_precision = 0

            val_pred_labels = []
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device).float()
                labels = labels.to(device).float()

                outputs = model(images)
                outputs = torch.squeeze(outputs, dim=1)

                val_loss += criterion(outputs, labels).item()

                accuracy, precision, recall, f1_score = calculate_metrics(
                    outputs, labels
                )
                val_accuracy += accuracy.item()
                val_precision += precision.item()
                val_recall += recall.item()
                val_f1_score += f1_score.item()

                cm = calculate_confusion_matrix(outputs, labels)
                val_cm = val_cm.to(cm.device)
                val_cm += cm

                val_roc_auc += calculate_roc_auc(outputs, labels)
                val_average_precision += calculate_average_precision(outputs, labels)

                val_pred_labels.extend(torch.sigmoid(outputs).cpu().tolist())

            val_loss /= len(val_loader)
            val_accuracy /= len(val_loader)
            val_precision /= len(val_loader)
            val_recall /= len(val_loader)
            val_f1_score /= len(val_loader)
            val_roc_auc /= len(val_loader)
            val_average_precision /= len(val_loader)

            val_cm = val_cm.reshape(-1)
            print(f"Confusion Matrix:\n{val_cm}")

            print(
                f"Validation: Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f} Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1_score:.4f}, ROC-AUC: {val_roc_auc:.4f}, Average Precision: {val_average_precision:.4f}"
            )

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), name + ".pt")

                val_pred_df = pd.DataFrame(
                    {
                        "path": val_df["path"].tolist(),
                        "predicted_label": val_pred_labels,
                    }
                )
                val_pred_df.to_csv(f"{name}.csv", index=False)

            writer.add_scalar("Validation/Loss", val_loss, epoch + 1)
            writer.add_scalar("Validation/Accuracy", val_accuracy, epoch + 1)
            writer.add_scalar("Validation/Precision", val_precision, epoch + 1)
            writer.add_scalar("Validation/Recall", val_recall, epoch + 1)
            writer.add_scalar("Validation/F1-Score", val_f1_score, epoch + 1)
            writer.add_scalar("Validation/ROC-AUC", val_roc_auc, epoch + 1)
            writer.add_scalar(
                "Validation/Average Precision", val_average_precision, epoch + 1
            )
            writer.add_text("Confusion Matrix", str(val_cm.tolist()), epoch + 1)

    writer.flush()
    writer.close()
    print("Zakonshil ***!")


if __name__ == "__main__":
    config_path = "E:/kbtu_courses/diploma_project/src/diploma/models/liveness/model_training/config.yaml"
    log_dir = "E:/kbtu_courses/diploma_project/src/diploma/models/liveness/model_training/logs"

    train(config_path, log_dir)
