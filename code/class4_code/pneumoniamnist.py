# D 练习：用卷积神经网络判断儿童胸片是正常还是肺炎（注释精简版，代码和 train.py 完全相同）
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)

EPOCHS = 15     # 训练轮数
LR = 0.001      # 学习率


# 1. 读数据：64×64 灰度胸片，0 = 正常，1 = 肺炎
data = np.load('pneumoniamnist_64.npz')
train_images = torch.tensor(data['train_images']).float().unsqueeze(1) / 255
train_labels = torch.tensor(data['train_labels']).long().flatten()
test_images = torch.tensor(data['test_images']).float().unsqueeze(1) / 255
test_labels = torch.tensor(data['test_labels']).long().flatten()

print('训练图：', train_images.shape[0], '张')
print('测试图：', test_images.shape[0], '张')

train_set = torch.utils.data.TensorDataset(train_images, train_labels)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)


# 2. 模型：三层卷积 + 两层全连接
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),                    # 64 → 32

    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),                    # 32 → 16

    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),                    # 16 → 8

    nn.Flatten(),
    nn.Linear(4096, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def accuracy(images, labels):
    with torch.no_grad():
        scores = model(images)
        pred = scores.argmax(1)
    right = (pred == labels)
    return right.float().mean().item()


# 3. 训练
losses = []
train_accs = []
test_accs = []

for epoch in range(1, EPOCHS + 1):

    total_loss = 0

    for images, labels in train_loader:
        scores = model(images)              # 前向：算得分
        loss = loss_fn(scores, labels)      # 算损失
        optimizer.zero_grad()
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新权重
        total_loss += loss.item()

    losses.append(total_loss / len(train_loader))
    train_accs.append(accuracy(train_images, train_labels))
    test_accs.append(accuracy(test_images, test_labels))

    print(f'第 {epoch} 轮   损失 {losses[-1]:.3f}   '
          f'训练准确率 {train_accs[-1]:.0%}   测试准确率 {test_accs[-1]:.0%}')


# 4. 画图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(losses, 'o-')
plt.title('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(train_accs, 'o-', label='train')
plt.plot(test_accs, 'o-', label='test')
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend()

plt.savefig('results.png')
plt.show()
