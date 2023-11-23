from torch import nn
class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.pool0 =  nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.pool1 =  nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.pool2 =  nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.pool3 =  nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.pool4 =  nn.MaxPool2d(kernel_size=2, return_indices=True)
        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1),
            # nn.BatchNorm2d(4),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1),
            # nn.BatchNorm2d(2),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=1, out_channels=2, kernel_size=1),
            # nn.BatchNorm2d(2),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1),
            # nn.BatchNorm2d(4),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1),
            # nn.BatchNorm2d(8),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        # decoder (upsampling)
        self.upsample0 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.upsample1 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            )
        self.upsample2 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.upsample3 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            )
        self.upsample4 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, padding=1),
            )
    def forward(self, x):
        # encoder
        e0, indices_e0 = self.pool0(self.enc_conv0(x))
        e1, indices_e1 = self.pool1(self.enc_conv1(e0))
        e2, indices_e2 = self.pool2(self.enc_conv2(e1))
        e3, indices_e3 = self.pool3(self.enc_conv3(e2))
        e4, indices_e4 = self.pool4(self.enc_conv4(e3))
        # bottleneck
        b = self.bottleneck_conv(e4)
        # decoder
        d0 = self.dec_conv0(self.upsample0(e4, indices_e4))
        d1 = self.dec_conv1(self.upsample1(d0, indices_e3))
        d2 = self.dec_conv2(self.upsample2(d1, indices_e2))
        d3 = self.dec_conv3(self.upsample3(d2, indices_e1))
        d4 = self.dec_conv4(self.upsample4(d3, indices_e0))  # no activation
        return d4