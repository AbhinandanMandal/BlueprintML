"""Generator (Context Encoder) and Discriminator.

Layer definitions, tensor shapes and attribute names are identical to the notebook so that
``state_dict`` keys of notebook-trained checkpoints load without any mapping.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """AlexNet-style encoder: 3x128x128 -> 256x8x8."""

    def __init__(self):
        super().__init__()

        def block(cin, cout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, s, p, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True))

        self.net = nn.Sequential(
            block(3,   64,  7, 2, 3),            # 128 -> 64
            nn.MaxPool2d(3, 2, 1),               # 64  -> 32
            block(64,  192, 5, 1, 2),            # 32
            nn.MaxPool2d(3, 2, 1),               # 32  -> 16
            block(192, 384, 3, 1, 1),            # 16
            block(384, 256, 3, 1, 1),            # 16
            block(256, 256, 3, 1, 1),            # 16  (conv5)
            nn.MaxPool2d(3, 2, 1),               # 16  -> 8  (pool5)
        )

    def forward(self, x):
        return self.net(x)


class ChannelWiseFC(nn.Module):
    """Fully-connected layer applied independently per channel (paper, Sec. 3.1).

    Input/Output: (B, m, n, n). Params: m * n^4 (+ m * n^2 bias).
    """

    def __init__(self, channels, size):
        super().__init__()
        self.m, self.n2 = channels, size * size
        self.weight = nn.Parameter(torch.empty(channels, self.n2, self.n2))
        self.bias = nn.Parameter(torch.zeros(channels, self.n2))
        bound = 1 / self.n2 ** 0.5          # same scheme as nn.Linear
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):
        B, m, n, _ = x.shape
        # flatten each feature map
        x = x.reshape(B, m, self.n2)
        x = torch.einsum('bmi,mij->bmj', x, self.weight)   # per-channel FC
        x = x + self.bias
        return x.reshape(B, m, n, n)


class Decoder(nn.Module):
    """Channel-wise FC -> 1x1 conv -> 5 up-convolutions. 256x8x8 -> 3x128x128."""

    def __init__(self, channels=256, size=8):
        super().__init__()
        self.cw_fc = ChannelWiseFC(channels, size)
        # stride-1 conv across channels
        self.mix = nn.Conv2d(channels, channels, 1)

        def up(cin, cout):                                # fractional-stride conv (x2)
            return nn.Sequential(
                nn.ConvTranspose2d(cin, cout, 4, 2, 1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True))

        self.up = nn.Sequential(
            up(256, 128),                                 # 8   -> 16
            up(128, 64),                                  # 16  -> 32
            up(64, 64),                                   # 32  -> 64
            up(64, 32),                                   # 64  -> 128
            # 5th up-conv, stride 1 -> RGB
            nn.ConvTranspose2d(32, 3, 3, 1, 1),
            # matches [-1, 1] normalization
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.mix(self.cw_fc(z))
        return self.up(z)


class ContextEncoder(nn.Module):
    """Generator: masked image (B,3,128,128) -> full-size prediction (B,3,128,128) in [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, masked_img):
        return self.decoder(self.encoder(masked_img))


class Discriminator(nn.Module):
    """Real (x) vs. fake (F((1-M)*x)). 3x128x128 -> single raw logit."""

    def __init__(self):
        super().__init__()

        def block(cin, cout, bn=True):
            layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3,   64, bn=False),   # 128 -> 64
            block(64,  128),            # 64  -> 32
            block(128, 256),            # 32  -> 16
            block(256, 512),            # 16  -> 8
            block(512, 512),            # 8   -> 4
        )
        self.classifier = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))   # raw logit
