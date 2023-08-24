"""
Majority of code written by André Pacheco (pacheco.comp@gmail.com). 
https://github.com/lmlima/BRACIS2022-Exploring-Advances-for-SLD/tree/main
"""


import torch
import torchvision.transforms as transforms
from sklearn import preprocessing
from PIL import Image
import os 
from torch.utils import data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MyDataset (data.Dataset):
    """
    This is the standard way to implement a dataset pipeline in PyTorch. We need to extend the torch.utils.data.Dataset
    class and implement the following methods: __len__, __getitem__ and the constructor __init__
    """

    def __init__(self, yaml_args, image_paths, labels = None, meta_data = None, transform=None):
        """
        The constructor gets the images path and their respectively labels and meta-data (if applicable).
        In addition, you can specify some transform operation to be carry out on the images.

        It's important to note the images must match with the labels (and meta-data if applicable). For example, the
        imgs_path[x]'s label must take place on labels[x].

        Parameters:
        :param imgs_path (list): a list of string containing the image paths
        :param labels (list) a list of labels for each image
        :param meta_data (list): a list of meta-data regarding each image. If None, there is no information.
        Defaul is None.
        :param transform (torchvision.transforms.Compose): transform operations to be carry out on the images
        """

        super().__init__()
        self.image_paths = image_paths
        if yaml_args["run_type"] == "forward_infer":
            self.labels = None
        else:
            self.labels = labels

        if yaml_args["include_metadata"]:
            self.meta_data = meta_data
        else:
            self.meta_data = None
        # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll get an exception
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()
    
    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.image_paths)


    def __getitem__(self, item):
        """
        It gets the image, labels and meta-data (if applicable) according to the index informed in `item`.
        It also performs the transform on the image.

        :param item (int): an index in the interval [0, ..., len(img_paths)-1]
        :return (tuple): a tuple containing the image, its label and meta-data (if applicable)
        """

        image = Image.open(self.image_paths[item]).convert("RGB")

        # Applying the transformations
        image = self.transform(image)

        img_id = self.image_paths[item]#.split('/')[-1]#.split('.')[0]

        if self.meta_data is None:
            meta_data = []
        else:
            meta_data = self.meta_data[item]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return image, labels, meta_data, img_id


def split_and_load(yaml_args, transforms, label_encoder = None, metadata_stats_dicts = None):
    if yaml_args["run_type"] == "forward_infer":
        return get_data_loader(yaml_args, transforms, metadata_stats_dicts = metadata_stats_dicts)
    return get_data_loader(yaml_args, transforms, label_encoder = label_encoder)


def get_image_paths_labels(base_path, label_encoder = None, forward_infer = False):
    image_paths = []
    labels = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                labels.append(root.split("/")[-1])
    
    if forward_infer:
        return image_paths
    
    labels = label_encoder.transform(labels)
    return image_paths, labels


def get_image_metadata(image_paths, yaml_args, meta_stats_dicts = None):
    image_col = yaml_args["meta_image_name_column"]
    # if metadata statistics exist (forward inference)
    if meta_stats_dicts is None:
        meta_cols = yaml_args["meta_columns"]
    else:
        meta_cols = list(meta_stats_dicts["metadata_mean_dict"].keys())
    meta_data = pd.read_csv(yaml_args["metadata_csv"])
    metadata_array = meta_data[meta_cols].values
    length_of_meta_obs = len(metadata_array[0])
    nan_counter = 0
    image_meta = []
    for idx, image in enumerate(image_paths):
        try:
            image_name = image.split('/')[-1]#.replace(".png", "")
            values = meta_data[meta_data[image_col] == image_name][meta_cols].values[0]
            image_meta.append(values.tolist())

        except IndexError:
            print(image_name, "not present in metadata file:", image)
            nan_counter += 1
            image_meta.append([np.nan] * length_of_meta_obs)

    print("Number of images missing metadata:", nan_counter)
    return image_meta



def metadata_impute_and_normalize(df, mean_array, std_array):
    metadata_array = [np.array(meta) for meta in df["metadata"]]
    for i, meta in enumerate(metadata_array):
        nan_indices = np.isnan(meta)
        metadata_array[i][nan_indices] = mean_array[nan_indices]
        norm_meta = (meta - mean_array) / std_array
        metadata_array[i] = norm_meta
    df["metadata"] = list(metadata_array)



def get_data_loader (yaml_args, transform, label_encoder = None, metadata_stats_dicts = None):
    """
    This function gets a list og images path, their labels and meta-data (if applicable) and returns a DataLoader
    for these files. You also can set some transformations using torchvision.transforms in order to perform data
    augmentation. Lastly, params is a dictionary that you can set the following parameters:
    batch_size (int): the batch size for the dataset. If it's not informed the default is 30
    shuf (bool): set it true if wanna shuffe the dataset. If it's not informed the default is True
    num_workers (int): the number thread in CPU to load the dataset. If it's not informed the default is 0 (which


    :param imgs_path (list): a list of string containing the images path
    :param labels (list): a list of labels for each image
    :param meta_data (list, optional): a list of meta-data regarding each image. If it's None, it means there's
    no meta-data. Default is None
    :param transform (torchvision.transforms, optional): use the torchvision.transforms.compose to perform the data
    augmentation for the dataset. Alternatively, you can use the jedy.pytorch.utils.augmentation to perform the
    augmentation. If it's None, none augmentation will be perform. Default is None
    :param batch_size (int): the batch size. If the key is not informed or params = None, the default value will be 30
    :param shuf (bool): if you'd like to shuffle the dataset. If the key is not informed or params = None, the default
    value will be True
    :param num_workers (int): the number of threads to be used in CPU. If the key is not informed or params = None, the
    default value will be  4
    :param pin_memory (bool): set it to True to Pytorch preload the images on GPU. If the key is not informed or
    params = None, the default value will be True
    :return (torch.utils.data.DataLoader): a dataloader with the dataset and the chose params
    """

    if yaml_args["run_type"] == "forward_infer":
        images = get_image_paths_labels(yaml_args["path_to_image_folder"], forward_infer=True)
        if (yaml_args["include_metadata"]):
            meta_data = get_image_metadata(images, yaml_args, metadata_stats_dicts)
            full_data = pd.DataFrame({"images" : images, "metadata" : meta_data})
            print("Replacing nans and normalizing data.")
            metadata_impute_and_normalize(full_data, np.array(list(metadata_stats_dicts["metadata_mean_dict"].values())),  np.array(list(metadata_stats_dicts["metadata_std_dict"].values())))
            dt = MyDataset(yaml_args, full_data["images"], meta_data=full_data["metadata"], transform=transform)
        else:
            full_data = pd.DataFrame({"images" : images})
            dt = MyDataset(yaml_args, full_data["images"], transform=transform)
        
        return data.DataLoader(dataset=dt, batch_size=yaml_args["batch_size"], shuffle=False, num_workers=yaml_args["num_workers"],
                               pin_memory=yaml_args["pin_memory"])

    images, labels = get_image_paths_labels(yaml_args["path_to_image_folder"], label_encoder)

    if (yaml_args["include_metadata"]):
        meta_data = get_image_metadata(images, yaml_args)
        full_data = pd.DataFrame({"images" : images, "labels" : labels, "metadata" : meta_data})
    else:
        full_data = pd.DataFrame({"images" : images, "labels" : labels})

    print("Performing train-test-split.")
    train_df, temp_test_df = train_test_split(full_data, test_size= 1 - yaml_args["train_proportion"], random_state=yaml_args["train_test_split_seed"])
    test_df, val_df = train_test_split(temp_test_df, test_size=0.5, random_state=yaml_args["train_test_split_seed"]) 
   
    print("Length of train dataset:", len(train_df))
    print("Length of validation dataset:", len(val_df))
    print("Length of test dataset:", len(test_df))

    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    test_df = test_df.reset_index()

    if (yaml_args["include_metadata"]): 
        print("Replacing nans and normalizing data.")
        train_metadata_array = np.array([np.array(meta) for meta in train_df["metadata"]])
        has_nan = np.isnan(train_metadata_array).any(axis=1)
        rows_with_nan = train_metadata_array[has_nan]
        print(len(rows_with_nan), "rows from train set have nan values. Ignoring these rows in mean and std calculation.")

        train_metadata_mean = np.nanmean(train_metadata_array, axis=0)
        train_metadata_std = np.nanstd(train_metadata_array, axis=0)
        train_metadata_std[train_metadata_std == 0] = 1.0

        metadata_impute_and_normalize(train_df, train_metadata_mean, train_metadata_std)
        metadata_impute_and_normalize(test_df, train_metadata_mean, train_metadata_std)
        metadata_impute_and_normalize(val_df, train_metadata_mean, train_metadata_std)

        train_dataset = MyDataset(yaml_args, train_df["images"], train_df["labels"], train_df["metadata"], transform)
        test_dataset = MyDataset(yaml_args, test_df["images"], test_df["labels"], test_df["metadata"], transform)
        val_dataset = MyDataset(yaml_args, val_df["images"], val_df["labels"], val_df["metadata"], transform)
        mean_dict = dict(zip(yaml_args["meta_columns"], train_metadata_mean))
        std_dict = dict(zip(yaml_args["meta_columns"], train_metadata_std))
    
    else:
        train_dataset = MyDataset(yaml_args, train_df["images"], train_df["labels"], None, transform)
        test_dataset = MyDataset(yaml_args, test_df["images"], test_df["labels"], None, transform)
        val_dataset = MyDataset(yaml_args, val_df["images"], val_df["labels"], None, transform)

    #if yaml_args["include_metadata"]:
        #collate = custom_collate_fn
    #    collate = None
    #else:
    #    collate = None 
    
    # train test split 

    train = data.DataLoader(dataset=train_dataset, batch_size=yaml_args["batch_size"], shuffle=True, num_workers=yaml_args["num_workers"],
                               pin_memory=yaml_args["pin_memory"])#, collate_fn=collate)
    test = data.DataLoader(dataset=test_dataset, batch_size=yaml_args["batch_size"], shuffle=False, num_workers=yaml_args["num_workers"],
                            pin_memory=yaml_args["pin_memory"])#, collate_fn=collate)
    val = data.DataLoader(dataset=val_dataset, batch_size=yaml_args["batch_size"], shuffle=False, num_workers=yaml_args["num_workers"],
                               pin_memory=yaml_args["pin_memory"])#, collate_fn=collate)
    

    if yaml_args["include_metadata"]:
        return train, test, val, mean_dict, std_dict

    return train, test, val
"""
def custom_collate_fn(batch):
    #Ensures that the metadata is properly stacked and in order 
    #(in some cases, data can become vertically split across rows instead of horizontally otherwise)
    images, labels, meta_data, img_ids = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    meta_data_padded = torch.stack([torch.DoubleTensor(meta) for meta in meta_data])
    return images, labels, meta_data_padded, img_ids
"""