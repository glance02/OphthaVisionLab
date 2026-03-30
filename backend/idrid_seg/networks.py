import torch
import torch.nn.functional as F
import torch.nn as nn
p = 0.2
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class MSRNet(nn.Module):
    def __init__(self, in_=1, feature=32, p=3, out_channels=2):
        super(MSRNet, self).__init__()
        self.inchannel=in_
        self.down_conv1 = self._conv(in_, feature, stride=1)
        self.down_conv2 = self._conv(feature, feature*2, stride=1)
        self.down_conv3 = self._conv(feature*2, feature * 3, stride=1)

        self.layer1 = self.make_layer(ResidualBlock, feature,  p-2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, feature*2, p-1, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, feature*3, p, stride=1)

        self.up_conv3 = nn.Sequential(ResidualBlock(feature * 3, feature * 2, 1))
        self.up_conv2 = nn.Sequential(ResidualBlock(feature*2, feature, 1))
        self.up_conv1 = nn.Sequential(ResidualBlock(feature, out_channels, 1))

        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x2 = F.max_pool2d(x2, 2)
        x3 = self.down_conv3(x2)
        x3 = F.max_pool2d(x3, 2)

        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)

        x3_up = self.up_conv3(self.Upsample(x3))
        x2_up = self.up_conv2(x3_up+x2)
        out = self.up_conv1(self.Upsample(x2_up) + x1)

        return F.log_softmax(out, dim=1)
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        self.inchannel = channels
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def _conv(self, in_channel, out_channel, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    from thop import profile
    model = MSRNet().cuda()
    a = torch.randn(10, 1, 128, 128).cuda()
    b = model(a)
    print(b.shape)
    print(count_parameters(model))
    flops, params = profile(model, inputs=(a,))
    print("params = %.5f | Floats = %.5f" % ((params)/ (1000 ** 2), flops / (1000 ** 3)))