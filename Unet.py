import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalInstanceNorm2d(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.inst_norm = torch.nn.InstanceNorm2d(num_features, affine=False)
        self.embed = torch.nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, style_index):
        out = self.inst_norm(x)
        gamma, beta = self.embed(style_index).chunk(2, 1) # split
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, style_num):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.cin = ConditionalInstanceNorm2d(out_channels, style_num)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style_idx):
      out = self.relu(self.cin(self.conv1(x), style_idx))
      return self.relu(self.cin(self.conv2(out), style_idx))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_num):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, style_num)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x, style_num):
        skip_out = self.double_conv(x, style_num)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_num):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest') #, align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels, style_num)

    def forward(self, down_input, skip_input, style_num):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x, style_num)


class UNet(nn.Module):
    def __init__(self, style_num, out_classes=3):
        super(UNet, self).__init__()

        self.down_conv1 = DownBlock(3, 64, style_num)
        self.down_conv2 = DownBlock(64, 128, style_num)
        self.down_conv3 = DownBlock(128, 256, style_num)

        self.double_conv = DoubleConv(256, 512, style_num)

        self.up_conv3 = UpBlock(256 + 512, 256, style_num)
        self.up_conv2 = UpBlock(128 + 256, 128, style_num)
        self.up_conv1 = UpBlock(128 + 64, 64, style_num)

        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x, style_idx=None):
        x, skip1_out = self.down_conv1(x, style_idx)
        x, skip2_out = self.down_conv2(x, style_idx)
        x, skip3_out = self.down_conv3(x, style_idx)

        x = self.double_conv(x, style_idx)
        
        x = self.up_conv3(x, skip3_out, style_idx)
        x = self.up_conv2(x, skip2_out, style_idx)
        x = self.up_conv1(x, skip1_out, style_idx)
        x = self.conv_last(x)
        return x
