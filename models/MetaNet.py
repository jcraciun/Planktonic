"""
Code written by André Pacheco (pacheco.comp@gmail.com). 
https://github.com/lmlima/BRACIS2022-Exploring-Advances-for-SLD/tree/main
"""

import torch 
from torch import nn

class MetaNet(nn.Module):
    """
    Implementing the MetaNet approach
    Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis - https://ieeexplore.ieee.org/document/9098645
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(MetaNet, self).__init__()
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 1),
            nn.ReLU(),
            nn.Conv2d(middle_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_maps, metadata):
        metadata = torch.unsqueeze(metadata, -1)
        metadata = torch.unsqueeze(metadata, -1)
        x = self.metanet(metadata)
        x = x * feat_maps
        return x