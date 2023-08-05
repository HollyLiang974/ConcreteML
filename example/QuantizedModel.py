'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: QuantizedModel
@Author: Sakura
@Time: 2023/7/20 15:32
@Software: PyCharm
功能描述: 只量化权重，偏置和激活函数不量化
实现步骤:
结果：
'''
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import brevitas.nn as qnn
# from concrete.ml.torch.compile import compile_brevitas_qat_model
import time
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4)
        self.conv1 = qnn.QuantConv2d(1, 32, 3, weight_bit_width=4, bias=False)
        self.relu1 =  qnn.QuantReLU(bit_width=4)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = qnn.QuantConv2d(32, 64, 3, weight_bit_width=4, bias=False)
        self.relu2 = qnn.QuantReLU(bit_width=4)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linear1 = qnn.QuantLinear(64 * 5 * 5, 4096, weight_bit_width=4, bias=False)
        self.relu3 = qnn.QuantReLU(bit_width=4)
        self.linear2 = qnn.QuantLinear(4096, 10, weight_bit_width=4, bias=False)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.quant_inp(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.quant_inp(x)
        x = self.flatten(x)
        x = self.quant_inp(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x

net=model()
train_data = torchvision.datasets.MNIST('./data', train=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
print('training on: ', device)
start_time=time.time()
for epoch in range(10):
    print('epoch:%d' % epoch)
    acc = 0.0
    sum = 0.0
    loss_sum = 0
    for batch, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        acc += torch.sum(torch.argmax(output, dim=1) == target).cpu().item()
        sum += len(target)
        loss_sum += loss
        if batch % 100 == 0:
            print('\tbatch: %d, loss: %.4f' % (batch, loss))
    print('epoch: %d， acc: %.2f%%, loss: %.4f' % (epoch, 100 * acc / sum, loss_sum / len(train_loader)))
end_time=time.time()
print("训练时长",end_time-start_time)
torch.save(net.state_dict(), '../model/quantized_model.pkl')