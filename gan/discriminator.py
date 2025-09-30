from torch import nn

# Class pytorchun neureal network modülünü kalıtım alıyor.
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), # 28x28 image => 784 boyutunda düz bi vektör
            nn.Linear(28*28, 512), # parametre sayısı = sol*sağ (784x512)+512 tane parametre
            # Ölü Nöron
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256), # (512*256)+256 => parametre
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1), # (256*1) + 1 => 257 parametre
            nn.Sigmoid() # 0-1 arasında değer dön.
        )

    def forward(self, x):
        return self.net(x)

# bir resim verilecek (gerçek-sahte) -> bu resimi incele, 0,1 dön. 0->sahte 1->gerçek