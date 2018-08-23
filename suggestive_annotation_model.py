import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_1 = ConvBlock(in_channels, out_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv_2 = ConvBlock(out_channels // 4, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)), inplace=True)
        z = F.relu(self.bn2(self.conv2(z)), inplace=True)
        z = F.relu(self.bn3(self.conv3(z)), inplace=True)
        z = z + self.conv_residual(x)
        return z


class UpsamplingBlock(nn.Module):
    def __init__(self, out_channels_for_encoder, n_times_to_upsample):
        super().__init__()
        up_convs = []
        initial_in_channel = out_channels_for_encoder[n_times_to_upsample - 1]
        out_channels_for_conv_trans = [ch//4 for ch in out_channels_for_encoder[:n_times_to_upsample][::-1]]

        for i in range(n_times_to_upsample):
            if i == 0:
                up_convs.append(nn.ConvTranspose2d(initial_in_channel,
                                                   out_channels_for_conv_trans[i],
                                                   kernel_size=2, stride=2))
            else:
                up_convs.append(nn.ConvTranspose2d(out_channels_for_conv_trans[i - 1],
                                                   out_channels_for_conv_trans[i],
                                                   kernel_size=2, stride=2))
        self.up_convs = nn.ModuleList(up_convs)

    def forward(self, x):
        for up_conv in self.up_convs:
            x = up_conv(x)
        return x


# In[25]:


class IntermediateFeatureMap(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bottleneck_1 = Bottleneck(in_channels, bottleneck_channels, in_channels)
        self.bottleneck_2 = Bottleneck(in_channels, bottleneck_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)
        return x

class SuggestiveAnnotationModel(nn.Module):
    def __init__(self, n_input_channels, n_classes, out_channels=[256, 256, 512, 1024, 1024, 1024]):
        super().__init__()
        encoder_blocks = []
        decoder_blocks = []
        for i, output_channel in enumerate(out_channels):
            if i == 0:
                encoder_blocks.append(InputBlock(in_channels=n_input_channels, out_channels=output_channel))
            else:
                encoder_blocks.append(IntermediateFeatureMap(in_channels=out_channels[i - 1],
                                                             bottleneck_channels=output_channel // 4,
                                                             out_channels=output_channel))
            decoder_blocks.append(UpsamplingBlock(out_channels_for_encoder=out_channels, n_times_to_upsample=i + 1))
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.out_conv_1 = ConvBlock(in_channels=(out_channels[0]//4) * len(out_channels),
                                    out_channels=n_classes, kernel_size=3, stride=1, padding=1)
        self.out_conv_2 = ConvBlock(in_channels=n_classes, out_channels=n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_image_embedding=True):
        encoder_outputs = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)

        decoder_outputs = []
        for encoder_output, decoder_block in zip(encoder_outputs, self.decoder_blocks):
            decoder_output = decoder_block(encoder_output)
            decoder_outputs.append(decoder_output)

        concat = torch.cat(decoder_outputs, 1)
        out = self.out_conv_1(concat)
        out = self.out_conv_2(out)

        if with_image_embedding:
            *_, last_encoder_output = encoder_outputs
            last_encoder_output = last_encoder_output.view(*last_encoder_output.shape[:2], -1)
            image_embeddings = last_encoder_output.mean(-1)
            return out, image_embeddings
        else:
            return out

model = SuggestiveAnnotationModel(n_input_channels=3, n_classes=3).cuda()
inp = torch.rand((2, 3, 512, 512)).cuda()
out = model(inp)


