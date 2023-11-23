from torch.nn import Module, ReLU, Conv2d, BatchNorm2d, Upsample, MaxPool2d
from torch import tensor, cat
from torch.nn.functional import pad


class VGGBlock(Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = ReLU(inplace=True)
        self.conv1 = Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = BatchNorm2d(middle_channels)
        self.conv2 = Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Up(Module):
    """Upscaling and concat"""

    def __init__(self):
        super().__init__()
        self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = tensor([x2.size()[2] - x1.size()[2]])
        diffX = tensor([x2.size()[3] - x1.size()[3]])

        x1 = pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = cat([x2, x1], dim=1)
        return x
