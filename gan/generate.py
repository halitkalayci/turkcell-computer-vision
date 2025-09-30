import torch
import torchvision
from generator import Generator
import os

Z_DIM = 100
CHECKPOINT_PATH = "./checkpoints/checkpoint_epoch_350.pth"
OUTPUT_DIR = "./"
NUM_IMAGES = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

# Generator modelini oluştur ve cihaza taşı
G = Generator().to(device)

# Checkpoint'i yükle
print(f"'{CHECKPOINT_PATH}' checkpoint'i yükleniyor...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

# Modelin state_dict'ini yükle
G.load_state_dict(checkpoint['G_state_dict'])

# Modeli değerlendirme moduna al
G.eval()

print("Resimler oluşturuluyor...")
# Rastgele gürültü oluştur
noise = torch.randn(NUM_IMAGES, Z_DIM, device=device)

# Gürültüden resimler oluştur
with torch.no_grad():
    fake_images = G(noise)

# Resimleri kaydet
output_path = os.path.join(OUTPUT_DIR, "generated.png")
torchvision.utils.save_image(fake_images, output_path, nrow=8, normalize=True)

print(f"{NUM_IMAGES} adet resim başarıyla oluşturuldu ve '{output_path}' dosyasına kaydedildi.")

