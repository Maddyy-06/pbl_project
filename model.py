import torch
import torch.nn as nn

class SurgeonUNet(nn.Module):
    def __init__(self):
        super(SurgeonUNet, self).__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)
            )
        self.enc1 = block(1, 64)
        self.enc2 = block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec = block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.up(e2)
        d = torch.cat([d, e1], dim=1)
        return torch.sigmoid(self.final(self.dec(d)))