import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.pad = nn.ZeroPad2d((0, 0, 1, 1))

        self.conv1 = nn.Conv2d(1, 50, kernel_size=(7, 1))
        self.bn1 = nn.BatchNorm2d(50)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))


        self.conv2 = nn.Conv2d(50, 50, kernel_size=(7, 2))
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(50 * 29 * 1, 256)
        self.fc2 = nn.Linear(256, 80)
        self.fc3 = nn.Linear(80, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool1(self.leaky_relu(self.bn1(self.conv1(x))))

        x = self.pad(x)
        x = self.pool2(self.leaky_relu(self.bn2(self.conv2(x))))

        x = self.flatten(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, (1, 128, 2)))




