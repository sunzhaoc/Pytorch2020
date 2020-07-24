"""
https://www.bilibili.com/video/BV1WT4y177SA?from=search&seid=5537861487238463993
"""
# 1 Load Datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 2 Define hyper-parameter
BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000

# 3 Construct pipeline
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 4 Download Datasets
from torch.utils.data import DataLoader

train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)

test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


# 5 Net
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(input_size, -1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


# Define Optimizer
model = Digit().to(DEVICE)

optimizer = optim.Adam(model.parameters())


# 7 Define Train Method
def train_model(model, device, train_loader, optimizer, epoch):
    # Train Model
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Set Gradient to 0
        optimizer.zero_grad()

        output = model(data)
        # Caulate Loss
        loss = F.cross_entropy(output, target)
        # Find the Index max
        pred = output.max(1, keepdim=True)
        loss.backward()

        optimizer.step()
        if (batch_index % 30) == 0:
            print("Train Epoch : {} \t Loss : {:.6f}".format(epoch, loss.item()))


def test_model(model, device, test_loader):
    # Model Validation
    model.eval()
    # Accurancy and Loss
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Test data
            output = model(data)
            # Caculate Loss
            test_loss += F.cross_entropy(output, target).item()
            # find max index
            pred = output.max(1, keepdim=True)[1]  # 0-Value 1-Key
            # Accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test -- Average Loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss,
                                                                          100.0 * correct / len(test_loader.dataset)))


for epoch in range(1, EPOCHS+1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)