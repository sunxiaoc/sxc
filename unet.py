import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True)
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)  # 逆卷积
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # 双线性插值

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2]-inputs1.size()[2]
        padding = 2*[offset//2, offset//2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unet(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, in_channels=3, is_deconv=True, is_batchnorm=True):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x/self.feature_scale) for x in filters]  # feature_scale=4,filters=[16,32,64,128,256]

        # downsample
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # unsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv(without and concat)
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final


if __name__ == "__main__":

    filepath1 = "C:/Users/29967/Desktop/多视角融合/程序训练/间隔10d数据-视电阻率矩阵/dipole/"
    filepath2 = "C:/Users/29967/Desktop/多视角融合/程序训练/间隔10d数据-视电阻率矩阵/wenner/"
    filepath3 = "C:/Users/29967/Desktop/多视角融合/程序训练/间隔10d数据-视电阻率矩阵/二极/"
    pathdir1 = os.listdir(filepath1)
    pathdir2 = os.listdir(filepath2)
    pathdir3 = os.listdir(filepath3)
    pathdir1.sort(key= lambda x:int(x[:-5]))  # 倒数第5位为分界
    pathdir2.sort(key= lambda x:int(x[:-5]))
    pathdir3.sort(key= lambda x:int(x[:-5]))
    for i in range(len(pathdir1)):
        d1 = np.array(pd.read_excel(filepath1 + pathdir1[i]))
        d2 = np.array(pd.read_excel(filepath2 + pathdir2[i]))
        d3 = np.array(pd.read_excel(filepath3 + pathdir3[i]))
        d = np.array([d1, d2, d3])


    model = unet(feature_scale=1)
    summary(model.cuda(), (3, 572, 572))



