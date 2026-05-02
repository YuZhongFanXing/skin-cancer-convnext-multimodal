import os
import torch

# 硬件配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径配置 — 将 ISIC 2019 数据集放入 data/ 目录即可
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
METADATA_PATH = os.path.join(DATA_DIR, "ISIC_2019_Training_Metadata.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "ISIC_2019_Training_Input")

# 超参数配置
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 3e-4
NUM_CLASSES = 8
REGION_SIZE = 4
PATIENCE = 8
WARMUP_EPOCHS = 3

# 输出目录
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL_DIR = os.path.join(RESULT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 将在运行时由数据预处理模块动态确定
META_DIM = None
