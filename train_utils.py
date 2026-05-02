import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from config import device, MODEL_DIR, PATIENCE, WARMUP_EPOCHS, NUM_EPOCHS


def create_optimizer(model, lr, weight_decay):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)))


def create_scheduler(optimizer):
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    attn_weights_list = []

    for images, metas, labels in tqdm(loader, desc="Training", leave=False):
        images, metas, labels = images.to(device), metas.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with autocast():
                outputs, attn_weights = model(images, metas)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, attn_weights = model(images, metas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        attn_weights_list.append(attn_weights.cpu().detach())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    avg_attn = torch.cat(attn_weights_list, dim=0).mean(dim=0)
    return epoch_loss, epoch_acc, avg_attn


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    all_logits = []
    attn_weights_list = []

    with torch.no_grad():
        for images, metas, labels in tqdm(loader, desc="Validation", leave=False):
            images, metas, labels = images.to(device), metas.to(device), labels.to(device)
            outputs, attn_weights = model(images, metas)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy())
            attn_weights_list.append(attn_weights.cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    avg_attn = torch.cat(attn_weights_list, dim=0).mean(dim=0)
    return epoch_loss, epoch_acc, all_preds, all_labels, all_logits, avg_attn


def save_model(model, optimizer, epoch, val_acc, path, class_weights=None, label_encoder=None):
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    if class_weights is not None:
        save_dict['class_weights'] = class_weights
    if label_encoder is not None:
        save_dict['label_encoder'] = label_encoder
    torch.save(save_dict, path)