import torch
import torch.nn as nn
from torchvision import models
from mixstyle import MixStyle # 引入 MixStyle

class FeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, use_mixstyle=False):
        super().__init__()
        self.use_mixstyle = use_mixstyle
        
        # if backbone == 'resnet18':
        #     self.backbone = models.resnet18(pretrained=pretrained)
        #     feature_dim = 512
        # elif backbone == 'resnet50':
        #     self.backbone = models.resnet50(pretrained=pretrained)
        #     feature_dim = 2048
        # elif backbone == 'vgg16':
        #     self.backbone = models.vgg16(pretrained=pretrained)
        #     feature_dim = 4096
        # else:
        #     raise ValueError(f"Unsupported backbone: {backbone}")
        
        # # 移除分类层
        # if backbone.startswith('resnet'):
        #     self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # elif backbone.startswith('vgg'):
        #     self.backbone = nn.Sequential(*list(self.backbone.features.children()))
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError("MixStyle baseline currenly supports resnet50")

        # 将 ResNet 拆解开，以便插入 MixStyle
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # 插入 MixStyle 层
        if self.use_mixstyle:
            self.mixstyle1 = MixStyle(p=0.5, alpha=0.1, mix='crossdomain')
            self.mixstyle2 = MixStyle(p=0.5, alpha=0.1, mix='crossdomain')
        
        # self.feature_dim = feature_dim
        # self.backbone_name = backbone
    
    def forward(self, x):
        # if self.backbone_name.startswith('resnet'):
        #     features = self.backbone(x)
        #     features = features.view(features.size(0), -1)
        # elif self.backbone_name.startswith('vgg'):
        #     features = self.backbone(x)
        #     features = self.avgpool(features)
        #     features = features.view(features.size(0), -1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if self.use_mixstyle:
            x = self.mixstyle1(x) # 混合一次风格
            
        x = self.layer2(x)
        if self.use_mixstyle:
            x = self.mixstyle2(x) # 再混合一次风格
            
        x = self.layer3(x)
        x = self.layer4(x)
        
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)

        return features

class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

class DomainAdapter(nn.Module):
    # 在初始化中增加 use_mixstyle 参数
    def __init__(self, config, use_mixstyle=False):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            backbone=config.backbone,
            pretrained=True,
            use_mixstyle=use_mixstyle # 传入标志位
        )
        self.classifier = Classifier(
            feature_dim=self.feature_extractor.feature_dim,
            num_classes=config.num_classes
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features