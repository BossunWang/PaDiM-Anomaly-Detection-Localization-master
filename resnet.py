import torch
import torch.nn as nn
from torchvision.models import resnet18, wide_resnet50_2

class CustomResnet(nn.Module):
    def __init__(self, model_type, pretrained=True):
        super().__init__()

        if model_type == 'resnet18':
            model = resnet18(pretrained=pretrained)
        elif model_type == 'wide_resnet50_2':
            model = wide_resnet50_2(pretrained=True, progress=True)
        else:
            print("model_type not exist")
            exit(1)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return [x1, x2, x3]


if __name__ == '__main__':
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResnet(pretrained=True).to(device)

    for i in range(5):
        input = torch.randn((16, 3, 224, 224), requires_grad=True).to(device)
        start_time = time.time()
        outputs = model(input)
        end_time = time.time()
        print(f'inference time:{end_time - start_time}')
        [print(out.shape) for out in outputs]