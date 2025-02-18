import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random

# 1. 加载 MNIST 数据集
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="D:\Project\BO\Incremental_Learning\data",
    train=True,
    download=True,
    transform=transform,
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(
    root="D:\Project\BO\Incremental_Learning\data",
    train=False,
    download=True,
    transform=transform,
)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)


# 2. 定义神经网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 增量学习：逐步加载数据
def incremental_train(model, trainset, batch_size=1000, epochs=1):
    model.train()
    dataset_size = len(trainset)
    num_splits = dataset_size // batch_size
    indices = list(range(dataset_size))
    random.shuffle(indices)  # 随机打乱数据顺序

    for i in range(num_splits):
        subset_indices = indices[i * batch_size : (i + 1) * batch_size]
        subset = torch.utils.data.Subset(trainset, subset_indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)

        for epoch in range(epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        print(f"Incremental Step {i+1}/{num_splits}, Loss: {loss.item():.4f}")


# 3. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 重新初始化模型
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 增量训练
incremental_train(model, trainset, batch_size=5000, epochs=1)

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"增量学习模型的准确率: {100 * correct / total:.2f}%")
