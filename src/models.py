import torch
import torch.nn as nn
from torchvision import models
import timm  


class ClassifierHead(nn.Module):
    def __init__(self, in_features=2048, num_classes=4, dropout_rate=0.5):
        """
        Args:
            in_features: Number of input features from base model (default 2048)
            num_classes: Number of output classes (default 4)
            dropout_rate: Dropout probability (default 0.5)
        """
        super(ClassifierHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
            
        )
    
    def forward(self, x):
        return self.classifier(x)


class DiagCovidPNA(nn.Module):
    def __init__(self, base_model_name='xception', num_classes=4, 
                 pretrained=True, dropout_rate=0.5):
        """
        Args:
            base_model_name: One of 'xception', 'inception_v3', 'inception_resnet_v2'
            num_classes: Number of output classes
            pretrained: Use ImageNet pre-trained weights
            dropout_rate: Dropout probability in classifier head
        """
        super(DiagCovidPNA, self).__init__()
        
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        
        # Load base model and get feature size
        if base_model_name == 'xception':
            self.base_model, in_features = self._create_xception(pretrained)
        elif base_model_name == 'inception_v3':
            self.base_model, in_features = self._create_inception_v3(pretrained)
        elif base_model_name == 'inception_resnet_v2':
            self.base_model, in_features = self._create_inception_resnet_v2(pretrained)
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Create custom classifier head
        self.classifier = ClassifierHead(
            in_features=in_features,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    
    def _create_xception(self, pretrained=True):
        """
        Create Xception base model.
        Output features: 2048 (after Global Average Pooling)
        """

        model = timm.create_model('xception', pretrained=pretrained)
        model.fc = nn.Identity()  # Remove final classification layer
        in_features = 2048
        
        return model, in_features
    
    def _create_inception_v3(self, pretrained=True):
        """
        Create Inception-v3 base model.
        Output features: 2048 (after Global Average Pooling)
        """
     
        if pretrained:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        model = models.inception_v3(weights=weights, init_weights=False)
        
        model.aux_logits = False
        model.AuxLogits = None
        
        model.fc = nn.Identity()
        model.dropout = nn.Identity()
        in_features = 2048
        
        return model, in_features
    
    def _create_inception_resnet_v2(self, pretrained=True):
        """
        Create Inception-ResNet-v2 base model.
        Output features: 1536 → Global Average Pooling
        """
        
        model = timm.create_model('inception_resnet_v2', pretrained=pretrained)
        model.classif = nn.Identity()
        model.head_drop = nn.Identity()
        in_features = 1536
        
        return model, in_features
    
    def forward(self, x):
        """
        Forward pass through the model. returns logits.
        """
        features = self.base_model(x)
        output = self.classifier(features)
        return output
    
    def predict(self, x, return_probs=True):

        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        
        if return_probs:
            return probabilities
        else:
            return torch.argmax(probabilities, dim=1)


def create_model(model_name='xception', num_classes=4, pretrained=True, dropout_rate=0.5):
    """
    Factory function to create a DiagCovidPNA model.
    Args:
        model_name: One of 'xception', 'inception_v3', 'inception_resnet_v2'
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
        dropout_rate: Dropout probability in classifier head
    
    Returns:
        DiagCovidPNA model instance
    """

    model = DiagCovidPNA(
        base_model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    return model


    


