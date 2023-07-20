'''
-*- ecoding: utf-8 -*-
@Enviroment: ConcreteML
@ModuleName: mnist
@Author: Sakura
@Time: 2023/7/20 16:27
@Software: PyCharm
功能描述: minist数据集的训练
实现步骤:
结果：
'''
import torch, torchvision
import torch.nn as nn

# 构建模型
net = nn.Sequential(
    nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2),

    # nn.Flatten(会将参数拉成二维,即(batch_size, ?))
    nn.Flatten(),
    nn.Linear(64 * 5 * 5, 4096), nn.ReLU(),
    nn.Linear(4096, 10)
)

train_data = torchvision.datasets.MNIST('./data', train=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
print('training on: ', device)
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