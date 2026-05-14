import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
from config import IMAGE_DIR, IMG_SIZE

warnings.filterwarnings('ignore', category=UserWarning)


class ISICDataPreprocessor:
    def __init__(self, gt_path, meta_path):
        self.gt_df = pd.read_csv(gt_path)
        self.meta_df = pd.read_csv(meta_path)

        # 合并数据
        self.df = pd.merge(self.gt_df, self.meta_df, on='image', how='inner')
        self._preprocess_labels()
        self._preprocess_metadata()

    def _preprocess_labels(self):
        lesion_types = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
        self.df['target'] = self.df[lesion_types].idxmax(axis=1)
        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df['target'])

        # 计算类别权重
        class_counts = self.df['label'].value_counts().sort_index().values
        self.class_weights = 1.0 / class_counts
        self.class_weights = self.class_weights / self.class_weights.sum()

    def _preprocess_metadata(self):

        # 处理缺失值
        age_median = self.df['age_approx'].median()
        self.df['age_approx'] = self.df['age_approx'].fillna(age_median)
        self.df[anatomy_col] = self.df[anatomy_col].fillna('unknown')
        self.df['sex'] = self.df['sex'].fillna('unknown')

        # 标准化数值特征
        age_mean, age_std = self.df['age_approx'].mean(), self.df['age_approx'].std()
        self.df['age_norm'] = (self.df['age_approx'] - age_mean) / age_std

        # 类别特征编码
        sex_dummies = pd.get_dummies(self.df['sex'], prefix='sex')
        site_dummies = pd.get_dummies(self.df[anatomy_col], prefix='site')
        self.meta_features = pd.concat([self.df['age_norm'], sex_dummies, site_dummies], axis=1)
        return self.meta_features.values.astype(np.float32)

    def get_data(self):
        train_df, val_df = train_test_split(
            self.df, test_size=0.2, stratify=self.df['label'], random_state=42
        )
        return train_df, val_df, self.meta_features.values.astype(np.float32)


# 图像变换
def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


class ISICDataset(Dataset):
    def __init__(self, df, meta_features, image_dir, transform=None, is_train=True):
        self.df = df
        self.meta_features = meta_features
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        self.image_names = df['image'].values
        self.labels = df['label'].values
        self.indices = df.index.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = None

        # 尝试不同扩展名读取图像
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
            img_path = os.path.join(self.image_dir, f"{img_name}{ext}")
            if os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as f:
                        img_data = np.frombuffer(f.read(), dtype=np.uint8)
                        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if image is not None:
                        break
                except Exception:
                    continue

        # 如果图像读取失败，创建空白图像
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # 处理图像通道
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # 应用变换
        if self.transform:
            try:
                image = self.transform(image)
            except:
                image = transforms.ToTensor()(image)

        # 获取元数据
        meta_idx = self.indices[idx]
        metadata = torch.tensor(self.meta_features[meta_idx])
        if self.is_train:
            metadata += torch.randn_like(metadata) * 0.05

        # 标签
        label = torch.tensor(self.labels[idx])
        return image, metadata, label
