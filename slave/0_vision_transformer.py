import os
import urllib.request
from urllib.error import HTTPError

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

import numpy as np
import pickle
from pprint import pprint
from rosie.log import logger
logger.info("vision transformer")

plt.set_cmap("cividis")
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()

# %load_ext tensorboard
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/VisionTransformers/")

L.seed_everything(42)  # For reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

###### ========================================================================
###### ------ Download pretrained models ------ ######
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/"
pretrained_files = [
    "tutorial15/ViT.ckpt",
    "tutorial15/tensorboards/ViT/events.out.tfevents.ViT",
    "tutorial5/tensorboards/ResNet/events.out.tfevents.resnet",
]
# 존재여부확인후 다운로드
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
print(f"Directory '{CHECKPOINT_PATH}' created or already exists.")

for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name.split("/", 1)[1])
    logger.info(f"Checking {file_path}")
    if "/" in file_name.split("/", 1)[1]:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(f"Something went wrong.\n {e}")
    else:
        print("File already exists.")

###### ========================================================================
# 평균 표준편차 계산

# CIFAR-10 데이터셋 로드
# http://www.cs.toronto.edu/~kriz/cifar.html
# cifar-10-batches-py는 CIFAR-10 데이터셋의 Python 버전
# 10개의 클래스 (비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭)
# 로 구성된 60,000개의 32x32 컬러 이미지로 이루어져 있습니다.
# 이 데이터셋은 머신 러닝 알고리즘을 훈련하고 평가하는 데 널리 사용
# batches.meta: 데이터셋에 대한 메타데이터
# data_batch_1: 훈련 데이터의 첫 번째 배치
# ~
# data_batch_5: 훈련 데이터의 다섯 번째 배치
# test_batch: 테스트 데이터

# 파일 경로 설정
# meta_file_path = os.path.join(DATASET_PATH,'cifar-10-batches-py/batches.meta')
# with open(meta_file_path, 'rb') as file:
#     meta_data = pickle.load(file, encoding='bytes')
# pprint(meta_data)

#train_dataset = CIFAR10(root='data',  train=True,  download=True, transform=transforms.ToTensor())
train_dataset = CIFAR10(root='data',  train=True,  transform=transforms.ToTensor())
# /workspace/git/sinbakai/slave/data/cifar-10-python.tar.gz

# 데이터셋의 모든 이미지를 순회하면서 각 채널(RGB)의 평균과 표준편차 계산
# 이를통해 preprocessing에서 이미지정규화시 사용할  평균과 표준편차 값을 얻을 수 있습니다.

#이미지를 정규화한다는 것은
#이미지의 픽셀값을 특정 범위로 변환하여 데이터의 분포를 일정하게 만드는 과정을 의미
# 안정적이고 빠른 학습을 위해 (분포의 일관성, 수렴속도향상, 수치안정성)
mean = 0.0
std = 0.0
num_samples = 0

# test sixx
# first_img, first_lbl = train_dataset[0]
# plt.imshow(np.transpose(first_img.numpy(), (1, 2, 0))) #(C, H, W)에서 1인 H를 첫번째로, 2인 W를 두번재로 0인 C를 3 번째로 이동시킴
# plt.title(f'Label: {first_lbl}')
# plt.show()
# first_img.mean([1, 2]), first_img.std([1, 2])

# data는(이미지, 레이블)튜플, data[0]은 이미지 data[1]은 레이블
for data in train_dataset:
    image = data[0]    # 3x32x32 (C, H, W)
    mean += image.mean([1, 2]) # H(높이), W(너비)차원을 제외한 모든 차원의 평균
    std  += image.std([1, 2])
    num_samples += 1   # 처리한 이미지의 수를 증가

mean /= num_samples
std /= num_samples

print(f'Mean: {mean}')  # RGB 채널별 평균 tensor([0.4914, 0.4822, 0.4465])
print(f'Std: {std}')    # RGB 채널별 표준편차 tensor([0.2470, 0.2435, 0.2616])

# ====== Step 1: Data Transformations =========================================
# Load the pretrained model
# 정규화 변환 정의
# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)
# ])

# 학습시, test_tranform과 비교하여 augmentation 추가.
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.49139968, 0.48215841, 0.44653091],
            [0.24703223, 0.24348513, 0.26158784]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.49139968, 0.48215841, 0.44653091],
            std=[0.24703223, 0.24348513, 0.26158784]
        ),
    ]
)
# ======= Step 2: Train-Test Split ============================================
train_dataset= CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
L.seed_everything(42)
train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])

# validation set에는 augmentation를 적용되지 않은 test_transform을 사용
val_dataset  = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform,  download=True)
L.seed_everything(42)
_, val_set   = torch.utils.data.random_split(val_dataset, [45000, 5000])

test_set     = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

# Define a set of data loaders
# DataLoader는 dataset을 batch로 나누어서 학습시키기 위한 iterator를 제공하여,
# 모델 학습시 데이터를 미니배치로 나누어서 학습시킬 수 있도록
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True,  drop_last=True,  pin_memory=True, num_workers=4)
val_loader   = torch.utils.data.DataLoader(val_set,   batch_size=128, shuffle=False, drop_last=False, num_workers=4)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=128, shuffle=False, drop_last=False, num_workers=4)

# =======  ============================================
# Visualize some examples
NUM_IMAGES = 4
CIFAR_images = torch.stack([val_set[idx][0] for idx in range(NUM_IMAGES)], dim=0)
img_grid = torchvision.utils.make_grid(CIFAR_images, nrow=4, normalize=True, pad_value=0.9)
img_grid = img_grid.permute(1, 2, 0)   # tensor shape: [C, H, W] -> [H, W, C] 재배열

plt.figure(figsize=(8, 8))
plt.title("Image examples of the CIFAR10 dataset")
plt.imshow(img_grid)
plt.axis("off")
plt.show()
plt.close()

# =============================================================================
# Image Classification with Vision Transformers
# =============================================================================
# 원래 Transformer는 sequence-to-sequence 모델로, sequence를 입력으로 받아 sequence를 출력
# time-step의 Input feature vector에 position encoding을 추가하여 sequence의 순서를 모델에 전달하면
# Transformer모델이 스스로 sequence의 순서를 학습할 수 있음

# 이미지는 2차원 공간정보를 가지고 있고,
# Tranformer는 sequence를 입력으로 받기 때문에
# 이미지를 sequence로 변환하기 위해, 이미지를 patch로 나누어서 sequence로 변환
# 즉, 이미지는 작은 patch들의 시퀀스로 생각함.
# 각 patch는 "word", "token"으로 생각하여, feature space에 투영하여 Transformer에 입력으로 사용
# 여기에 "position encoding을 추가"하여 이미지의 공간정보를 유지하고,
# classification을 위해 마지막에 "CLS token을 추가"하여 분류를 수행

def img_to_patch(x, patch_size, flatten_channels=True):
    """
    이미지를 작은 patch로 분할하고, 필요에 따라 flatten하는 함수
    이를 통해 이미지를 patch로 나누어서 Transformer에 입력으로 사용할 수 있습니다.
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W] (batch, channels, height, width)
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    # patch크기에 맞게 x텐서를 재구성.
    # 이미지공간정보 더잘 캡춰하기 위해 하나의 이미지를 여러개 patch로 쪼갬
    x = x.reshape(B, C,    H//patch_size, patch_size,    W//patch_size, patch_size)
    x = x.permute(0, 2, 4,   1, 3, 5)  # [B, H',  W',   C, p_H, p_W]
    x = x.flatten(1, 2)                # [B,  H'*W',    C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x
# reshape 함수
# 텐서의 데이터를 유지하면서 새로운 형태로 변경
# 텐서의 차원을 재구성하여 새로운 형태로 반환합니다.
# 데이터의 순서를 변경하지 않습니다.

# permute 함수
# 텐서의 차원을 재배열합니다.
# 텐서의 차원을 지정된 순서에 따라 재배열하여 새로운 텐서를 반환합니다.
# 데이터의 순서를 변경합니다.

# Original Tensor: [2, 3, 4]
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],

#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])

# Reshaped Tensor (3, 2, 4): 결과 텐서는 [3, 2, 4] 형태입니다.
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7]],

#         [[ 8,  9, 10, 11],
#          [12, 13, 14, 15]],

#         [[16, 17, 18, 19],
#          [20, 21, 22, 23]]])

# Permuted Tensor (1, 0, 2): 1, 0, 2는 새로운 차원의 순서, 결과 텐서는 [3, 2, 4] 형태입니다.
# tensor([[[ 0,  1,  2,  3],
#          [12, 13, 14, 15]],

#         [[ 4,  5,  6,  7],
#          [16, 17, 18, 19]],

#         [[ 8,  9, 10, 11],
#          [20, 21, 22, 23]]])
# =============================================================================
# CIFAR_images: torch.Size([3, 3, 32, 32])    batch_size,              channels,     height,      width
# img_patches: torch.Size([3, 64, 3, 4, 4])   batch_size, num_patches, channels, patch_size, patch_size
img_patches = img_to_patch(CIFAR_images, patch_size=4, flatten_channels=False)

fig, ax = plt.subplots(CIFAR_images.shape[0], 1, figsize=(14, 3))
fig.suptitle("Images as input sequences of patches")
for i in range(CIFAR_images.shape[0]):
    img_grid = torchvision.utils.make_grid(img_patches[i], nrow=64, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)
    ax[i].imshow(img_grid)
    ax[i].axis("off")
plt.show()
plt.close()


# =============================================================================
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0] #query, attention weight
        x = x + self.linear(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out

class ViT(L.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

#  대규모 이미지 분류 벤치마크(ImageNet...)이 아닌
#  고전적인 소규모 벤치마크(CIFAR10..)에서도 Vision Transformer가 잘 적용되는지 실험

# 이를 위해 CIFAR10에 대해, 처음부터 Vision Transformer를 훈련해 보자
# 우선 Pytorch lighting 모듈을 위한 훈련함수
# 이 함수를 통해, 이미 다운로드해 둔 pre-trained model 로딩

def train_model(**kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"),
        accelerator="auto",
        devices=1,
        max_epochs=180,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = ViT.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = ViT(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result


model, results = train_model(
    model_kwargs={
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "patch_size": 4,
        "num_channels": 3,
        "num_patches": 64,
        "num_classes": 10,
        "dropout": 0.2,
    },
    lr=3e-4,
)
print("ViT results", results)

import os
import subprocess
log_dir = "saved_models"
subprocess.run(["tensorboard", "--logdir", log_dir])