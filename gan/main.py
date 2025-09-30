import torch
from torch.utils.data import dataloader
import torchvision
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

if __name__ == "__main__":
    dataset = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    G = generator.Generator().to(device)
    D = discriminator.Discriminator().to(device)
    
    loss_fn = torch.nn.BCELoss()
    optim_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999)) #TODO: betas nedir? momentum etkisi? 
    optim_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.999))
    
    # 16 adet, 100 boyutlu gürültü.
    fixed_noise = torch.randn(16, Z_DIM, device=device)
    
    from matplotlib import pyplot as plt
    
    def save_images(images: torch.Tensor, epoch: int, out_dir=SAMPLES_DIR, nrow=4):
        grid = torchvision.utils.make_grid(images, nrow=nrow)
        npimg = grid.detach().cpu().numpy().transpose((1,2,0))
        plt.imshow(npimg)
        plt.axis("off")
        out_path = os.path.join(out_dir, f"epoch_{epoch}.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    
    for epoch in range(EPOCHS):
        for real, _ in dataloader:
            real = real.to(device)
            bsz = real.size(0)
    
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)
    
            ## Discriminator Eğit 
            noise = torch.randn(bsz, Z_DIM, device=device)
            fake_images = G(noise)
    
            D_real = D(real)
            D_fake = D(fake_images.detach())
    
            D_loss_real = loss_fn(D_real, real_labels)
            D_loss_fake = loss_fn(D_fake, fake_labels)
            D_loss = D_loss_real + D_loss_fake
    
            optim_D.zero_grad()
            D_loss.backward()
            optim_D.step()
            ## Generator Eğit
            output = D(fake_images)
            G_loss = loss_fn(output, real_labels)
    
            optim_G.zero_grad()
            G_loss.backward()
            optim_G.step()
        if (epoch + 1) % SAVE_IMG_EVERY == 0:
            G.eval()
            with torch.no_grad():
                fake = G(fixed_noise)
            save_images(fake.detach(), epoch)
            G.train()
