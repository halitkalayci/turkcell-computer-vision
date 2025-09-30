import torch
from torch.utils.data import dataloader
import generator
import discriminator
import os 
from torchvision import datasets, transforms
print(torch.version.cuda)
print(torch.cuda.is_available())

# python -m venv venv
# .\venv\Scripts\activate
# python -m pip install --upgrade pip
# pip3 install torch torchvision

# Ayarlar
BATCH_SIZE = 128
EPOCHS = 350
LR_G = 2e-4
LR_D = 2e-4
Z_DIM = 100

SAVE_IMG_EVERY = 10
SAVE_CHECKPOINT_EVERY = 50

DATA_ROOT = "./data"
CKPT_DIR = "./checkpoints"
SAMPLES_DIR = "./generated_fake_imgs"

SEED = 42
torch.manual_seed(SEED)
#

device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # [-1,1]
])

dataset = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)