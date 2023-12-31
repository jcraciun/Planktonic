"""
Majority of code written by André Pacheco (pacheco.comp@gmail.com). 
https://github.com/lmlima/BRACIS2022-Exploring-Advances-for-SLD/tree/main
"""

import torch
from torch import nn
from models import MetaBlock
from models import MetaNet 
import warnings


class MyEffnet (nn.Module):

    def __init__(self, efnet, num_class, neurons_reducer_block=256, p_dropout=0.5,
                 comb_method=None, comb_config=None, n_feat_conv=1280):

        super(MyEffnet, self).__init__()
        _n_meta_data = 0
        if comb_method is not None:
            if comb_config is None:
                raise Exception("You must define the comb_config since you have comb_method not None")

            if comb_method == 'metablock':
                if isinstance(comb_config, int):
                    raise Exception("comb_config must be a list/tuple to define the number of feat maps and the metadata")
                print("Warning: in metablock, ensure comb_config values is a factor of n_feat_conv divided by number of feature maps and hyperparameters.")
                print("n_feat_conv divided by feature maps and hyperparams is:", n_feat_conv / comb_config[0] / comb_config[2])
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
                print("Warning: in metanet, ensure product of feature maps, hyperparam1, and hyperparam2 is equal to n_feat_conv.")
                print("This product is:", comb_config[2] * comb_config[3] * comb_config[4])
                print("n_feat_conv is:", n_feat_conv)
                self.comb = MetaNet.MetaNet(comb_config[0], comb_config[1], comb_config[2]) # (n_meta, middle, feature_maps)
                self.comb_feat_maps = comb_config[2]
                self.hyper_param1 = comb_config[3]
                self.hyper_param2 = comb_config[4]
            else:
                raise Exception("There is no comb_method called " + comb_method + ". Please, check this out.")
        else:
            self.comb = None

        self.efnet = efnet
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Feature reducer
        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            if comb_method == 'concat':
                warnings.warn("You're using concat with neurons_reducer_block=0. Make sure you're doing it right!")
            self.reducer_block = None

        # Here comes the extra information (if applicable)
        if neurons_reducer_block > 0:
            self.classifier = nn.Linear(neurons_reducer_block + _n_meta_data, num_class)
        else:
            self.classifier = nn.Linear(n_feat_conv + _n_meta_data, num_class)


    def forward(self, img, meta_data=None):

        # Checking if when passing the metadata, the combination method is set
        if meta_data is not None and self.comb is None:
            raise Exception("There is no combination method defined but you passed the metadata to the model!")
        if meta_data is None and self.comb is not None:
            raise Exception("You must pass meta_data since you're using a combination method!")

        x = self.efnet.extract_features(img)
        x = self.avg_pooling(x)

        if self.comb == None:
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif self.comb == 'concat':
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block. In this case, it must be defined
            x = torch.cat([x, meta_data], dim=1)  # concatenation
        elif isinstance(self.comb, MetaBlock):
            x = x.view(x.size(0), self.comb_feat_maps, self.hyper_param, -1).squeeze(-1)  # getting the feature maps
            x = self.comb(x, meta_data.float())  # applying MetaBlock
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block
        elif isinstance(self.comb, MetaNet):
            x = x.view(x.size(0), self.comb_feat_maps, self.hyper_param1, self.hyper_param2).squeeze(-1)  # getting the feature maps
            x = self.comb(x, meta_data.float())  # applying metanet
            x = x.view(x.size(0), -1)  # flatting
            if self.reducer_block is not None:
                x = self.reducer_block(x)  # feat reducer block

        return self.classifier(x)