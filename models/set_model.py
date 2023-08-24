"""
Majority of code written by André Pacheco (pacheco.comp@gmail.com). 
https://github.com/lmlima/BRACIS2022-Exploring-Advances-for-SLD/tree/main
"""

from models import VIT
from models import gmtNet
from models import effNet
from transformers import ViTModel
from load_plankton import utils
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet


def set_model (model_name, num_class, neurons_reducer_block=0, comb_method=None, comb_config=None, pretrained=True,
         freeze_conv=True, p_dropout=0.5, transforms_mean=None, transforms_std=None):

    if pretrained:
        pre_ptm = 'imagenet'
        pre_torch = True
    else:
        pre_torch = False
        pre_ptm = None
    model = None
    
    if model_name == "gmtNet":
        model = gmtNet.gmtNet(num_class, comb_method=comb_method, comb_config=comb_config, p_dropout=p_dropout)
        model_transforms = transforms.Compose([
            utils.PadCenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] ) if transforms_mean is None
            else transforms.Normalize(mean=transforms_mean,
            std=transforms_std)
    ])
    if model_name == 'vit':
        pretrained_vit = 'google/vit-base-patch16-224-in21k' # Default
        # pretrained_vit = "facebook/dino-vitb16" # Interpretability
        # pretrained_vit = 'vit_large_patch16_384'  # Performance
        # pretrained_vit = "google/vit-huge-patch14-224-in21k" # Too large, memory issues
        model = VIT.MyViT(ViTModel.from_pretrained(pretrained_vit), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)
        model_transforms = transforms.Compose([
            utils.PadCenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5] ) if transforms_mean is None
            else transforms.Normalize(mean=transforms_mean,
            std=transforms_std)
    ])
    if model_name == 'efficientnet-b0':
        model = effNet.MyEffnet(EfficientNet.from_pretrained(model_name), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)
        model_transforms = transforms.Compose([
            utils.PadCenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) if transforms_mean is None
            else transforms.Normalize(mean=transforms_mean,
            std=transforms_std)
    ])

       
    return model, model_transforms