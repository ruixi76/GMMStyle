import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'vgg16':
            self.backbone = models.vgg16(pretrained=pretrained)
            feature_dim = 4096
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 移除分类层
        if backbone.startswith('resnet'):
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone.startswith('vgg'):
            self.backbone = nn.Sequential(*list(self.backbone.features.children()))
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_dim = feature_dim
        self.backbone_name = backbone
    
    def forward(self, x):
        if self.backbone_name.startswith('resnet'):
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        elif self.backbone_name.startswith('vgg'):
            features = self.backbone(x)
            features = self.avgpool(features)
            features = features.view(features.size(0), -1)
        
        return features

class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

class DomainAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            backbone=config.backbone,
            pretrained=True
        )
        self.classifier = Classifier(
            feature_dim=self.feature_extractor.feature_dim,
            num_classes=config.num_classes
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features