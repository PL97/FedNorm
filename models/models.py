import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict


class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)
        return x


class DigitModel_G(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, num_groups=2, **kwargs):
        super(DigitModel_G, self).__init__()

        if num_groups == -1:
            self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
            self.bn1 = nn.InstanceNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
            self.bn2 = nn.InstanceNorm2d(num_groups, 64)
            self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
            self.bn3 = nn.InstanceNorm2d(128)
        
            self.fc1 = nn.Linear(6272, 2048)
            self.bn4 = nn.InstanceNorm2d(2048)
            self.fc2 = nn.Linear(2048, 512)
            self.bn5 = nn.InstanceNorm2d(512)
            self.fc3 = nn.Linear(512, num_classes)

        else:
            self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
            self.bn1 = nn.GroupNorm(num_groups, 64)
            self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
            self.bn2 = nn.GroupNorm(num_groups, 64)
            self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
            self.bn3 = nn.GroupNorm(num_groups, 128)
        
            self.fc1 = nn.Linear(6272, 2048)
            self.bn4 = nn.GroupNorm(num_groups, 2048)
            self.fc2 = nn.Linear(2048, 512)
            self.bn5 = nn.GroupNorm(num_groups, 512)
            self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        # x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        # x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
            
                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class AlexNet_G(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10, num_groups=10):
        super(AlexNet_G, self).__init__()

        if num_groups == -1:
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                    ('bn1', nn.InstanceNorm2d(64)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                    ('bn2', nn.InstanceNorm2d(192)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                    ('bn3', nn.InstanceNorm2d(384)),
                    ('relu3', nn.ReLU(inplace=True)),

                    ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                    ('bn4', nn.InstanceNorm2d(256)),
                    ('relu4', nn.ReLU(inplace=True)),

                    ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                    ('bn5', nn.InstanceNorm2d(256)),
                    ('relu5', nn.ReLU(inplace=True)),
                    ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ])
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            self.classifier = nn.Sequential(
                OrderedDict([


                    ('clf_fc1', nn.Linear(256 * 6 * 6, 4096)),
                    # ('clf_bn6', nn.InstanceNorm1d(4096)),
                    ('clf_relu6', nn.ReLU(inplace=True)),
                    ('clf_fc2', nn.Linear(4096, 4096)),
                    # ('clf_bn7', nn.InstanceNorm1d(4096)),
                    ('clf_relu7', nn.ReLU(inplace=True)),
                    ('clf_fc3', nn.Linear(4096, num_classes)),
                ])
            )

        else:
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                    ('bn1', nn.GroupNorm(num_groups, 64)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                    ('bn2', nn.GroupNorm(num_groups, 192)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                    ('bn3', nn.GroupNorm(num_groups, 384)),
                    ('relu3', nn.ReLU(inplace=True)),

                    ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                    ('bn4', nn.GroupNorm(num_groups, 256)),
                    ('relu4', nn.ReLU(inplace=True)),

                    ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                    ('bn5', nn.GroupNorm(num_groups, 256)),
                    ('relu5', nn.ReLU(inplace=True)),
                    ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ])
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            self.classifier = nn.Sequential(
                OrderedDict([
                    ('clf_fc1', nn.Linear(256 * 6 * 6, 4096)),
                    # ('clf_bn6', nn.GroupNorm(num_groups, 4096)),
                    ('clf_relu6', nn.ReLU(inplace=True)),

                    ('clf_fc2', nn.Linear(4096, 4096)),
                    # ('clf_bn7', nn.GroupNorm(num_groups, 4096)),
                    ('clf_relu7', nn.ReLU(inplace=True)),
                    ('clf_fc3', nn.Linear(4096, num_classes)),
                ])
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def generate_models(args):
    if args.group:
        model = AlexNet_G(num_classes=10, num_groups=int(args.num_groups))
    else:
        model = AlexNet(num_classes=10)
    return model