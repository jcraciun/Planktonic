"""
Majority of code written by Jeff Ellen (ellen@spawar.navy.mil). 
"""

import torch
from torch import nn
from models import MetaBlock
from models import MetaNet 

class gmtNet(nn.Module):
    r"""Model with metadata for 128 x 128 images"""
    # n_feat_cov = 512
    def __init__(self, num_class, comb_method=None, comb_config=None, p_dropout=0.5):
        super(gmtNet, self).__init__()

        _n_meta_data = 0
        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if isinstance(comb_config, int):
                    raise Exception("comb_config must be a list/tuple to define the number of feat maps and the metadata")
                print("Warning: in metablock, ensure comb_config values is a factor of n_feat_conv divided by number of metadata features.")
                print("n_feat_conv divided by number of metadata features is:", 512 / comb_config[1])
                self.comb = MetaBlock.MetaBlock(comb_config[0], comb_config[1]) # (feature maps, metadata)
                self.comb_feat_maps = comb_config[0]
                self.hyper_param = comb_config[2] # hyperparam
            elif comb_method == 'concat':
                if not isinstance(comb_config, int):
                    raise Exception("comb_config must be int for 'concat' method")
                _n_meta_data = comb_config
                self.comb = 'concat'
            elif comb_method == 'metanet':
                if isinstance(comb_config, int):
                    raise Exception("comb_config must be a list/tuple to define the number of feat maps and the metadata")
                self.comb = MetaNet.MetaNet(comb_config[0], comb_config[1], comb_config[2]) # (n_meta, middle, feature_maps)
                self.comb_feat_maps = comb_config[2]
                self.hyper_param1 = comb_config[3]
                self.hyper_param2 = comb_config[4]
            else:
                raise Exception("There is no comb_method called " + comb_method + ". Please, check this out.")
        else:
            self.comb = None
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_x = nn.Linear(512, 256)
        self.fc_gmt = nn.Linear(_n_meta_data, 256)

        self.fc_1 = nn.Sequential(
            nn.Linear(256 + 256, 512), nn.ReLU(), nn.Dropout(p=p_dropout)
        )
        self.fc = nn.Sequential(nn.Linear(512, num_class))

        # x.requires_grad = True for fc layers with metadata (from or dealing with)
        # loss per batch instead of epoch 
    def forward(self, out, meta_data=None):
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.avgpool(out)

        if self.comb == None:
            out = out.reshape(out.size(0), -1) # flattening
        elif self.comb == 'concat':
            out = out.view(out.size(0), -1) 
            out = self.fc_x(out)
            meta_data = self.fc_gmt(meta_data)
            out = torch.cat((out, meta_data), 1)
        elif isinstance(self.comb, MetaBlock.MetaBlock):
            out = out.view(out.size(0), self.comb_feat_maps, self.hyper_param, -1).squeeze(-1) # getting the feature maps
            out = self.comb(out, meta_data.float()) # applying MetaBlock
            out = out.view(out.size(0), -1) # flatting
        elif isinstance(self.comb, MetaNet.MetaNet):
            #print(out.size())
            out = out.view(out.size(0), self.comb_feat_maps, self.hyper_param1, self.hyper_param2).squeeze(-1)  # getting the feature maps
            out = self.comb(out, meta_data.float())  # applying metanet
            out = out.view(out.size(0), -1)  # flatting

        out = self.fc_1(out)
        out = self.fc(out)
        return out
        