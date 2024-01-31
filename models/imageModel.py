from torch import nn
from torchvision import models


class ImageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.full_resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(
            *list(self.full_resnet.children())[:-1],
            nn.Flatten(),
        )

        # self.hidden_linear = nn.Sequential(
        #     nn.Dropout(config['img_dropout']),
        #     nn.Linear(self.full_resnet.fc.in_features, config['hidden_dim']),
        #     nn.ReLU(inplace=True),
        # )

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.full_resnet.fc.in_features, config['hidden_dim']),
            nn.ReLU(inplace=True),
        )

        for param in self.full_resnet.parameters():
            if config['finetune']:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, imgs, attention=False):
        feature = self.resnet(imgs)
        if attention:
            hidden_feature = self.hidden_linear(feature)
            return hidden_feature, feature
        return self.linear(feature)
