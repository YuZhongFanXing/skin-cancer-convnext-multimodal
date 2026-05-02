import torch
import torch.nn as nn
import timm
from config import REGION_SIZE, NUM_CLASSES


class SpatialPyramidProjection(nn.Module):
    def __init__(self, in_channels, out_channels, grid_size=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_regions = grid_size * grid_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return x


class CrossModalAttentionPooling(nn.Module):
    def __init__(self, img_feat_dim, meta_feat_dim, hidden_dim=256):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(img_feat_dim + meta_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(img_feat_dim + meta_feat_dim, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, img_regions, meta_feat):
        B, num_regions, _ = img_regions.shape
        meta_expanded = meta_feat.unsqueeze(1).expand(-1, num_regions, -1)
        combined = torch.cat([img_regions, meta_expanded], dim=-1)

        attn_scores = self.attention_net(combined).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_features = torch.einsum('brd,br->bd', combined, attn_weights)

        fused_feat = self.fc(weighted_features)
        fused_feat = self.norm(fused_feat)
        fused_feat = self.activation(fused_feat)
        fused_feat = self.dropout(fused_feat)

        return fused_feat, attn_weights


class MultiModalConvNeXt(nn.Module):
    def __init__(self, num_classes, meta_dim, region_size=4, region_feat_dim=256):
        super().__init__()
        # 图像分支
        self.img_backbone = timm.create_model(
            'convnext_tiny', pretrained=True, num_classes=0, features_only=True, out_indices=[3]
        )
        backbone_out_dim = 768

        # 空间金字塔投影
        self.spatial_pyramid = SpatialPyramidProjection(
            backbone_out_dim, region_feat_dim, grid_size=region_size)

        # 元数据分支
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2))

        # 跨模态注意力池化
        self.cross_attn_pooling = CrossModalAttentionPooling(
            region_feat_dim, 128, hidden_dim=384)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes))

    def forward(self, img, meta):
        img_features = self.img_backbone(img)[0]
        img_regions = self.spatial_pyramid(img_features)
        meta_features = self.meta_fc(meta)
        fused_features, attn_weights = self.cross_attn_pooling(img_regions, meta_features)
        logits = self.classifier(fused_features)
        return logits, attn_weights