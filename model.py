"""Model definitions for domain adaptation."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class FeatureExtractor(nn.Module):
    """
    Feature extractor backbone with optional MixStyle integration.
    
    Supports ResNet50 architecture with MixStyle layers inserted after
    layer1 and layer2 for style mixing augmentation.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        use_mixstyle: bool = False
    ):
        super().__init__()
        self.use_mixstyle = use_mixstyle
        
        if backbone != 'resnet50':
            raise ValueError(
                "MixStyle baseline currently only supports resnet50"
            )
        
        resnet = models.resnet50(pretrained=pretrained)
        self.feature_dim = 2048
        
        # Decompose ResNet for MixStyle insertion
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Insert MixStyle layers if enabled
        if self.use_mixstyle:
            from mixstyle import MixStyle
            self.mixstyle1 = MixStyle(p=0.5, alpha=0.1, mix='crossdomain')
            self.mixstyle2 = MixStyle(p=0.5, alpha=0.1, mix='crossdomain')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if self.use_mixstyle:
            x = self.mixstyle1(x)
        
        x = self.layer2(x)
        if self.use_mixstyle:
            x = self.mixstyle2(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        
        return features


class Classifier(nn.Module):
    """Simple linear classifier head."""
    
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify features into categories."""
        return self.fc(x)


class DomainAdapter(nn.Module):
    """
    Main domain adaptation model combining feature extractor and classifier.
    """
    
    def __init__(self, config, use_mixstyle: bool = False):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            backbone=config.backbone,
            pretrained=True,
            use_mixstyle=use_mixstyle
        )
        self.classifier = Classifier(
            feature_dim=self.feature_extractor.feature_dim,
            num_classes=config.num_classes
        )
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension from extractor."""
        return self.feature_extractor.feature_dim
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from input."""
        return self.feature_extractor(x)
    
    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """Classify extracted features."""
        return self.classifier(features)
    
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and features."""
        features = self.extract_features(x)
        logits = self.classify_features(features)
        return logits, features
