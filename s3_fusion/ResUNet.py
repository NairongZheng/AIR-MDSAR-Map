"""
This code is referenced from https://github.com/assassint2017/MICCAI-LITS2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import weights_init

# znr
BN_MOMENTUM = 0.01

class ResUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2 ,training=False):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        # znr
        self.hyper_channel = 8

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(in_channel, 16, 3, 1, padding=1),
            nn.PReLU(16),

            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=False),

            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        # znr
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel * 4 * self.hyper_channel,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0)
        )

    def forward(self, inputs):      # input[8, 1, 8, 256, 256]

        long_range1 = self.encoder_stage1(inputs) + inputs      # [8, 16, 8, 256, 256] + [8, 1, 8, 256, 256] = [8, 16, 8, 256, 256]

        short_range1 = self.down_conv1(long_range1)             # [8, 32, 4, 128, 128]

        long_range2 = self.encoder_stage2(short_range1) + short_range1      # [8, 32, 4, 128, 128]
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training)

        short_range2 = self.down_conv2(long_range2)             # [8, 64, 2, 64, 64]

        long_range3 = self.encoder_stage3(short_range2) + short_range2      # [8, 64, 2, 64, 64]
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training) # [8, 64, 2, 64, 64]

        short_range3 = self.down_conv3(long_range3)             # [8, 128, 1, 32, 32]

        long_range4 = self.encoder_stage4(short_range3) + short_range3      # [8, 128, 1, 32, 32]
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training) # [8, 128, 1, 32, 32]

        short_range4 = self.down_conv4(long_range4)             # [8, 256, 1, 32, 32]

        outputs = self.decoder_stage1(long_range4) + short_range4   # [8, 256, 1, 32, 32]
        outputs = F.dropout(outputs, self.dorp_rate, self.training) # [8, 256, 1, 32, 32]

        output1 = self.map1(outputs)                            # [8, classes, 8, 512, 512]   (上采样(8, 16, 16))

        short_range6 = self.up_conv2(outputs)                   # [8, 128, 2, 64, 64]

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6     # [8, 128, 2, 64, 64]
        outputs = F.dropout(outputs, self.dorp_rate, self.training)     # [8, 128, 2, 64, 64]

        output2 = self.map2(outputs)                            # [8, classes, 8, 512, 512]   (上采样(4, 8, 8))

        short_range7 = self.up_conv3(outputs)                   # [8, 64, 4, 128, 128]

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7     # [8, 64, 4, 128, 128]
        outputs = F.dropout(outputs, self.dorp_rate, self.training)         # [8, 64, 4, 128, 128]

        output3 = self.map3(outputs)                            # [8, classes, 8, 512, 512]   (上采样(2, 4, 4))

        short_range8 = self.up_conv4(outputs)                   # [8, 32, 8, 256, 256]

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.map4(outputs)

        # znr
        output = torch.cat([output1, output2, output3, output4], 1)
        b, c, ch, h, w = output.shape
        output = output.view(b, c * ch, h, w)
        output = self.last_layer(output)
        return output

        # if self.training is True:
        #     return output1, output2, output3, output4
        # else:
        #     return output4

if __name__ == '__main__':
    model = ResUNet()
    model.apply(weights_init.init_model)
    image = np.random.normal(0,1,(8,1,8,256,256)).astype(np.float32)
    image = torch.from_numpy(image)
    pred = model(image)
    pass

    