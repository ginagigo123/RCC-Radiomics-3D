import torch
import torch.nn as nn
import torch.nn.functional as F
import monai

def conv3x3x3_block(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(inplace=True)
    )

class DeepProfiler(nn.Module):
    def __init__(self,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2,
                 use_mask=True):
        super(DeepProfiler, self).__init__()

        self.in_planes = [16, 32, 64, 128]

        self.AttentionLayer = monai.networks.blocks.Convolution(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            conv_only=True
        )
        self.use_mask = use_mask

        if self.use_mask == False:
            self.AttentionLayer = None

        # in: 2, out: 16
        # first block
        self.conv1 = conv3x3x3_block(n_input_channels,
                                        self.in_planes[0])
        self.conv2 = conv3x3x3_block(self.in_planes[0],
                                     self.in_planes[0])
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        # second block
        self.conv3 = conv3x3x3_block(self.in_planes[0],
                                        self.in_planes[1])
        self.conv4 = conv3x3x3_block(self.in_planes[1],
                                        self.in_planes[1])
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        # third block
        self.conv5 = conv3x3x3_block(self.in_planes[1],
                                        self.in_planes[2])
        self.conv6 = conv3x3x3_block(self.in_planes[2],
                                        self.in_planes[2])
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=1)

        # fourth block
        self.conv7 = conv3x3x3_block(self.in_planes[2],
                                        self.in_planes[3])
        self.conv8 = conv3x3x3_block(self.in_planes[3],
                                        self.in_planes[3])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_radiomics = nn.Linear(128, 10)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x, ROI=None):
        x = self.conv1(x)
        if self.use_mask:
            AttentionMap = self.AttentionLayer(F.interpolate(ROI, size=x.shape[2:5], mode='trilinear'))
            x = x + AttentionMap

        x = self.conv2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        radiomics = self.fc_radiomics(x)
        x = self.fc(x)
        
        return x, radiomics


