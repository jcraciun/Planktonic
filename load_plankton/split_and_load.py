import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast 
import numpy as np
from load_plankton.create_dataloader import get_data_loader
from sklearn.model_selection import train_test_split


def split_and_load(yaml_args, transforms):
    # TO DO: GET RID OF THIS FILE (PUT THIS FUNCTION SOMEWHERE ELSE)
    if yaml_args["run_type"] == "forward_infer":
        return get_data_loader(yaml_args, transforms)
    
    print("Splitting data and creating data loaders")
    if yaml_args["include_metadata"]:
         train_dataloader, test_dataloader, val_dataloader = get_data_loader(yaml_args, transforms)
    else:
        train_dataloader, test_dataloader, val_dataloader = get_data_loader(yaml_args, transforms)

    return train_dataloader, test_dataloader, val_dataloader



