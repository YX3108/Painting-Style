import timm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv  #
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
import pandas as pd
import os
import gc
import itertools
import random
from tqdm import tqdm
import time

def train_and_evaluate(trainset, valset, epoch, batch_size, lr, factor, patience):

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class ImageFolderWithPaths(datasets.ImageFolder):
        def __getitem__(self, index):
            sample, target = super().__getitem__(index)
            path, _ = self.samples[index]
            return sample, target, path

    train_dataset = ImageFolderWithPaths(root=trainset, transform=train_transform)
    val_dataset = ImageFolderWithPaths(root=valset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)



    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    # model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    # model = timm.create_model('resnet101', pretrained=True, num_classes=num_classes)
    # model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=num_classes)
    # model = timm.create_model('efficientnet_b5', pretrained=True, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Model is on device:", next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_epochs = epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    best_val_loss = np.inf
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, labels, paths in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.update(1)
                pbar.set_postfix(loss=running_loss / (total // len(images)), accuracy=100. * correct / total)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total

        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, paths in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()


        val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    print('Finished Training')
    return best_val_loss, best_val_accuracy


trainset = 'Trainset'
valset   = 'valset'
epoch = 30
batch_size = 10

lr_list = [0.001, 0.005, 0.0001, 0.0005, 0.00001]
factor_list = [0.1, 0.5]
patience_list = [3, 5, 7]
param_grid = list(itertools.product(lr_list, factor_list, patience_list))
print(param_grid)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = f"gridsearch_swin_ReduceLROnPlateau_{timestamp}.csv"
CSV_HEADER = ["lr", "factor", "patience", "best_val_loss", "best_val_accuracy(%)", "time_sec"]

def append_result_row(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

in_memory_records = []

for idx, (lr, factor, patience) in enumerate(param_grid, start=1):
    print(f"[{idx}/{len(param_grid)}] START  lr={lr}, factor={factor}, patience={patience}")


    torch.cuda.empty_cache()
    gc.collect()

    start_t = time.time()
    try:
        best_val_loss, best_val_acc = train_and_evaluate(
            trainset=trainset,
            valset=valset,
            epoch=epoch,
            batch_size=batch_size,
            lr=lr,
            factor=factor,
            patience=patience,
        )
    except Exception as e:
        best_val_loss, best_val_acc = np.inf, -1.0
        print(f"[{idx}] ❗ERROR (lr={lr}, factor={factor}, patience={patience}) ：{e}")

    elapsed = time.time() - start_t

    row = {
        "lr": lr,
        "factor": factor,
        "patience": patience,
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy(%)": float(best_val_acc),
        "time_sec": round(elapsed, 2),
    }
    append_result_row(out_csv, row)
    in_memory_records.append(row)

    print(f"[{idx}/{len(param_grid)}] DONE   lr={lr}, factor={factor}, patience={patience} | "
          f"best_val_acc={best_val_acc:.2f}%, best_val_loss={best_val_loss:.6f}, time={elapsed:.2f}s")
