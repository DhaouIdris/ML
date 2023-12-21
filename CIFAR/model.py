import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
import operator
from torchvision import models



def linear(config, input_size, output_size):
    """
    congig      -- Configuration parameters from the yaml file
    input_size  -- tuple
    output_size -- int

    """

    layers = [
        nn.Flatten(start_dim=1),
        nn.Linear(reduce(operator.mul, input_size, 1), output_size),
    ]
    return nn.Sequential(*layers)

###  test wether the model is working or not
def test_linear():
    cfg = {"class": "Linear"}
    batch_size = 64
    input_size = (3, 32, 32)
    num_classes = 18
    model = linear(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")

class LinearNet(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.input_size = reduce(operator.mul, input_size, 1)
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class ConvNet(nn.Module):
    def __init__(self, config, input_size, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class ResNet18(torch.nn.Module):
    def __init__(self, config, input_size, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # Modify the first layer to accept 3-channel 32x32 input
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Modify the average pooling layer to handle smaller input
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Modify the output layer to match the number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_features, num_classes)  # CIFAR-100 has 100 classes

    def forward(self, x):
        return self.resnet(x)


class ResNet101(torch.nn.Module):
    def __init__(self, config, input_size, num_classes):
        super(ResNet101, self).__init__()
        self.resnet = models.resnet101(pretrained=True)

        # Modify the first layer to accept 3-channel 32x32 input
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Modify the average pooling layer to handle smaller input
        self.resnet.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Modify the output layer to match the number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_features, num_classes)  

    def forward(self, x):
        return self.resnet(x)


class VGG11(torch.nn.Module):
    def __init__(self, config, input_size, num_classes):
        super(VGG11, self).__init__()
        self.vgg = models.vgg11(pretrained=True)

        # Modify the first layer to accept 3-channel 32x32 input
        self.vgg.features[0] = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Modify the output layer to match the number of classes
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg(x)


class VGG16(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, output_dim))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def VanillaCNN(cfg, input_size, num_classes):
    layers = []
    cin = input_size[0]
    cout = 16
    for i in range(cfg["model"]["num_layers"]):
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(conv_relu_bn(cout, cout))
        layers.extend(conv_down(cout, 2 * cout))
        cin = 2 * cout
        cout = 2 * cout
    conv_model = nn.Sequential(*layers)

    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
    out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)]
    return nn.Sequential(conv_model, *out_layers)
