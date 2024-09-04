import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=(3,1), padding=(1,0), stride=1)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=(5,1), padding=(2,0), stride=1)
        self.p4_1 = nn.MaxPool2d(kernel_size=(3,1), stride=1, padding=(1,0))
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, Inception, num_classes):
        super(GoogLeNet, self).__init__()
        self.pad = nn.ZeroPad2d((0, 0, 1, 1))

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1,0)))

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=(1, 1)),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,1), padding=(1,0), stride=(1, 1)), #change
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1),padding=(1,0)))

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)))

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)))

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet(Inception).to(device)
    print(summary(model, (1, 128, 2)))
