import torch
import torch.nn as nn
import timm

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder="tf_efficientnet_b7_ns", dropout_rate=0.0):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=True)
        
        # Remove the classifier head
        if hasattr(self.encoder, 'classifier'):
            in_features = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
        elif hasattr(self.encoder, 'fc'):
            in_features = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, 'head'):
            in_features = self.encoder.head.in_features
            self.encoder.head = nn.Identity()
        else:
            # For models with different attribute names
            in_features = self.encoder.num_features
            
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(in_features, 1)
        
    def forward(self, x):
        features = self.encoder(x)
        features = self.dropout(features)
        output = self.classifier(features)
        return output