import torch
import torch.nn as nn
from torchsummary import summary
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from torch import optim
import torchvision

class DownBlock(nn.Module): #下采样
    def __init__(self, num_convs, inchannels, outchannels, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for i in range(num_convs):
            if i == 0:
                blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
            else:
                blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
            blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

    def forward(self, x):
        return self.layer(x)

class UpBlock(nn.Module): #上采样
    def __init__(self, inchannels, outchannels):
        super(UpBlock, self).__init__()
        self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.convt(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, nchannels=3, nclasses=1):
        super(UNet, self).__init__()
        self.down1 = DownBlock(2, nchannels, 64, pool=False)
        self.down2 = DownBlock(3, 64, 128)
        self.down3 = DownBlock(3, 128, 256)
        # self.down4 = DownBlock(3, 256, 512)
        # self.down5 = DownBlock(3, 512, 1024)
        # self.up1 = UpBlock(1024, 512)
        # self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.out = nn.Sequential(
            nn.Conv2d(64, nclasses, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # x4 = self.down4(x3)
        # x5 = self.down5(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        return self.out(x)

def train_net(net, device, dataset, epochs=500, batch_size=3, lr=0.00001):
    Loss = []
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.SmoothL1Loss().cuda()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for xx_train, yy_train in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = xx_train.to(device=device, dtype=torch.float32)
            label = yy_train.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)

            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'unet1_best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())
    Loss0 = np.array(Loss)
    np.save('C:/Users/29967/Desktop/多视角融合/程序训练/三合一输入unet结果/epoch_{}'.format(epochs),Loss0)



if __name__ == "__main__":
    import time
    start = time.perf_counter()
    # 加载数据
    filepath1 = "C:/Users/29967/Desktop/多视角融合/程序训练/数据2/"
    pathdir1 = os.listdir(filepath1)
    # pathdir1.sort(key=lambda x: int(x[:-4]))
    filepath2 = "C:/Users/29967/Desktop/多视角融合/程序训练/数据标签2/"
    pathdir2 = os.listdir(filepath2)
    # pathdir2.sort(key=lambda x: int(x[:-5]))
    x_train, x_test, y_train, y_test = train_test_split(pathdir1, pathdir2, test_size=0.2)
    r1 = []
    for file in x_train:
        d = np.load(filepath1 + file)
        r1.append(d)
    xx_train = np.array(r1)
    r2 = []
    for file in y_train:
        d = np.array(pd.read_excel(filepath2 + file))
        d = d.reshape(1, d.shape[0], d.shape[1])
        r2.append(d)
    yy_train = np.array(r2)
    r3 = []
    for file in x_test:
        d = np.load(filepath1 + file)
        r3.append(d)
    xx_test = np.array(r3)
    r4 = []
    for file in y_test:
        d = np.array(pd.read_excel(filepath2 + file))
        d = d.reshape(1, d.shape[0], d.shape[1])
        r4.append(d)
    yy_test = np.array(r4)
    xx_train = torch.tensor(xx_train)
    yy_train = torch.tensor(yy_train)
    dataset = Data.TensorDataset(xx_train, yy_train)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet().cuda()
    summary(net.cuda(), (3, 40, 160))
    net.to(device=device)
    # 开始训练
    train_net(net, device, dataset)
    end = time.perf_counter()
    print("运行时间为", round(end - start), 'seconds')
    # #ceshi
    # filepath3 = "C:/Users/29967/Desktop/多视角融合/程序训练/测试数据/"
    # pathdir3 = os.listdir(filepath3)
    # r5 = []
    # for file in pathdir3:
    #     d = np.load(filepath3 + file)
    #     r5.append(d)
    # test = np.array(r5)
    # test = torch.tensor(test)
    # test = test.to(device=device, dtype=torch.float32)
    # predict = net(test)
    #
    # # a = predict[0]
    # # a = a.cpu().detach().numpy()
    # # a = a.reshape(40,160)
    #
    # filepath4 = "C:/Users/29967/Desktop/多视角融合/程序训练/三合一输入unet结果/"
    # for i in range(9):
    #     a = predict[i]
    #     a = a.cpu().detach().numpy()
    #     a = a.reshape(40, 160)
    #     np.save(filepath4+pathdir3[i],a)





