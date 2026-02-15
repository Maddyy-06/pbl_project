# models/cnn_model.py
# Phase 3: ResNet18 transfer learning for tumor classification
# To be implemented in Months 3-5

import torch
import torch.nn as nn
import torchvision.models as models

class TumorClassifier(nn.Module):
    """ResNet18 fine-tuned for kidney tumor detection"""
    def __init__(self, num_classes=2):
        super(TumorClassifier, self).__init__()
        # TODO: Implement in Phase 3 (Month 3-5)
        # We will:
        # - Load pretrained ResNet18
        # - Replace last layer
        # - Fine-tune on KiTS19
        pass
    
    def forward(self, x):
        pass
    
    def extract_features(self, x):
        """Extract features for ensemble with VAE"""
        pass