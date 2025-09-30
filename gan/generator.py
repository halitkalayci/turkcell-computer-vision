from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256), # (100*256)+256 => 25700 tane parametre
            nn.ReLU(True),
            nn.Linear(256, 512), # (256*512)+512 => 131328 tane parametre
            nn.ReLU(True),
            nn.Linear(512, 1024), # (512*1024)+1024 => 524544 tane parametre
            nn.ReLU(True),
            nn.Linear(1024, 784), # (1024*784)+784 => 821248 tane parametre
            nn.Tanh() # -1 - 1 arası
        )
    
    def forward(self, x):
        return self.net(x).view(-1, 1, 28, 28)