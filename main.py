import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from config import device, GROUND_TRUTH_PATH, METADATA_PATH, IMAGE_DIR, \
    NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, \
    RESULT_DIR, MODEL_DIR, META_DIM, REGION_SIZE, PATIENCE, WARMUP_EPOCHS, NUM_EPOCHS
from data_preprocessing import ISICDataPreprocessor, get_transforms, ISICDataset
from model import MultiModalConvNeXt
from train_utils import create_optimizer, create_scheduler, train_epoch, validate, save_model, lr_lambda
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# 打印设备信息
print(f"Using device: {device}")

# 数据预处理
print("Preprocessing data...")
preprocessor = ISICDataPreprocessor(GROUND_TRUTH_PATH, METADATA_PATH)
train_df, val_df, all_meta_features = preprocessor.get_data()
META_DIM = all_meta_features.shape[1]  # 更新全局配置
print(f"Metadata dimension: {META_DIM}")

# 创建数据集
train_transform, val_transform = get_transforms()
train_dataset = ISICDataset(train_df, all_meta_features, IMAGE_DIR, train_transform, True)
val_dataset = ISICDataset(val_df, all_meta_features, IMAGE_DIR, val_transform, False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
)

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# 初始化模型
print("Initializing model...")
model = MultiModalConvNeXt(NUM_CLASSES, META_DIM, REGION_SIZE).to(device)

# 损失函数和优化器
class_weights_tensor = torch.tensor(preprocessor.class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
optimizer = create_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
scheduler = create_scheduler(optimizer)

# 混合精度
scaler = GradScaler() if device.type == 'cuda' else None

# 训练历史
history = {'train_loss': [], 'train_acc': [], 'train_attn': [],
           'val_loss': [], 'val_acc': [], 'val_attn': [], 'val_auc': []}

# 训练主循环
best_val_acc, no_improve_epochs = 0.0, 0
print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 60)

    # 学习率信息
    if epoch < WARMUP_EPOCHS:
        print(f"Warmup phase: Learning rate = {optimizer.param_groups[0]['lr']:.7f}")
    else:
        print(f"Cosine annealing: Learning rate = {optimizer.param_groups[0]['lr']:.7f}")

    # 训练和验证
    train_loss, train_acc, train_attn = train_epoch(
        model, train_loader, optimizer, criterion, device, scaler
    )
    val_loss, val_acc, val_preds, val_labels, val_logits, val_attn = validate(
        model, val_loader, criterion, device
    )
    scheduler.step()

    # 计算平均AUC
    val_logits = np.array(val_logits)
    val_labels_onehot = np.eye(NUM_CLASSES)[val_labels]
    auc_scores = roc_auc_score(val_labels_onehot, val_logits, average=None)
    avg_auc = np.mean(auc_scores)

    # 记录历史
    for key, value in zip(
            ['train_loss', 'train_acc', 'train_attn', 'val_loss', 'val_acc', 'val_attn', 'val_auc'],
            [train_loss, train_acc, train_attn, val_loss, val_acc, val_attn, avg_auc]
    ):
        history[key].append(value)

    # 打印统计信息
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {avg_auc:.4f}")

    # 早停和模型保存
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_epochs = 0
        save_model(
            model, optimizer, epoch + 1, val_acc,
            f"{MODEL_DIR}/best_model.pth",
            preprocessor.class_weights,
            preprocessor.label_encoder
        )
        print(f"Saved best model with val acc: {val_acc:.4f}")
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs}/{PATIENCE} epochs")
        if no_improve_epochs >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # 定期保存
    if (epoch + 1) % 5 == 0:
        save_model(
            model, optimizer, epoch + 1, val_acc,
            f"{MODEL_DIR}/checkpoint_epoch_{epoch + 1}.pth"
        )

# 保存最终模型
if no_improve_epochs < PATIENCE:
    save_model(
        model, optimizer, epoch + 1, val_acc,
        f"{MODEL_DIR}/final_model.pth"
    )

print("\nTraining completed!")

# ----------------------------------------
# 结果可视化和分析
# ----------------------------------------
class_names = preprocessor.label_encoder.classes_

# 1. 绘制训练历史
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 注意力权重热力图
plt.subplot(2, 2, 3)
attn_matrix = np.array([attn.numpy() for attn in history['train_attn']])
sns.heatmap(attn_matrix.T, cmap="YlGnBu", cbar_kws={'label': 'Attention Weight'})
plt.title('Training Attention Weights Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Region')
plt.yticks(ticks=np.arange(REGION_SIZE * REGION_SIZE) + 0.5,
           labels=[f"Region {i + 1}" for i in range(REGION_SIZE * REGION_SIZE)])

plt.subplot(2, 2, 4)
attn_matrix = np.array([attn.numpy() for attn in history['val_attn']])
sns.heatmap(attn_matrix.T, cmap="YlGnBu", cbar_kws={'label': 'Attention Weight'})
plt.title('Validation Attention Weights Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Region')
plt.yticks(ticks=np.arange(REGION_SIZE * REGION_SIZE) + 0.5,
           labels=[f"Region {i + 1}" for i in range(REGION_SIZE * REGION_SIZE)])

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/training_history.png")
plt.close()

# 2. 混淆矩阵
cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"{RESULT_DIR}/confusion_matrix.png")
plt.close()

# 3. 分类报告
report = classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"{RESULT_DIR}/classification_report.csv")

# 4. 注意力权重分布
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
final_train_attn = history['train_attn'][-1].numpy()
plt.bar(range(1, REGION_SIZE * REGION_SIZE + 1), final_train_attn, color='skyblue')
plt.title("Final Training Attention Distribution")
plt.xlabel("Region")
plt.ylabel("Attention Weight")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
final_val_attn = history['val_attn'][-1].numpy()
plt.bar(range(1, REGION_SIZE * REGION_SIZE + 1), final_val_attn, color='salmon')
plt.title("Final Validation Attention Distribution")
plt.xlabel("Region")
plt.ylabel("Attention Weight")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/attention_distribution.png")
plt.close()

# 5. 绘制验证集AUC曲线
plt.figure(figsize=(10, 6))
plt.plot(history['val_auc'], label='Validation AUC')
plt.title('Validation AUC Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.savefig(f"{RESULT_DIR}/validation_auc.png")
plt.close()

print("\nAnalysis completed! Results saved in 'result_chai4' directory.")