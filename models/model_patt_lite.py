import torch
import torch.nn as nn
from torchvision.ops.misc import ConvNormActivation
import torch.nn.functional as F

from models.MobileNet_v1 import MobileNetV1, DepthWiseSeparableConv2d
from models.attn import SelfAttn


class Model(nn.Module):
    def __init__(self, num_classes=7):
        super(Model, self).__init__()
        self.mobilenet_v1 = MobileNetV1(num_classes)
        self.pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0)
        self.separ_conv1 = DepthWiseSeparableConv2d(512, 256, 2)
        self.separ_conv2 = DepthWiseSeparableConv2d(256, 256, 2)
        self.point_conv = ConvNormActivation(256,
                                             256,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             norm_layer=nn.BatchNorm2d,
                                             activation_layer=nn.ReLU,
                                             inplace=True,
                                             # bias=False,
                                             )
        self.attention_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            SelfAttn(size_emb=256, n_classes=256, nhead=1),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x1 = self.mobilenet_v1(x)
        x1 = self.pad(x1)
        x2 = self.separ_conv1(x1)
        x3 = self.separ_conv2(x2)
        x4 = self.point_conv(x3)
        x4 = F.adaptive_avg_pool2d(x4, (1, 1)).squeeze(-1).squeeze(-1)
        out = self.attention_classifier(x4)

        return out


# from torchinfo import summary
#
# model = Model()
# summary(model, (1, 3, 224, 224))
