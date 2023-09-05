"""
Majority of code written by André Pacheco (pacheco.comp@gmail.com). 
https://github.com/lmlima/BRACIS2022-Exploring-Advances-for-SLD/tree/main

NOTE: Custom collate function is commented out, as it does not work with the current metadata implementation. 
If you make edits and the metadata is sorted across columns of the matrix instead of rows, uncomment to fix this issue.  

"""


import torchvision.transforms as transforms
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
        :param image_paths (list): a list of string containing the image paths
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
    """
    Splits the data into train-test-validate if relevant. 
    """
    if yaml_args["run_type"] == "forward_infer":
        # no need for split, requires metadata information from training to impute NaNs
        return get_data_loader(yaml_args, transforms, metadata_stats_dicts = metadata_stats_dicts)
    # requires label encoder
    return get_data_loader(yaml_args, transforms, label_encoder = label_encoder)


def get_image_paths_labels(base_path, label_encoder = None, forward_infer = False):
    """
    Walks though the base directory and returns the image paths and labels for each file with valid ending. 
    """
    image_paths = []
    labels = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
                # label is the name of the folder containing the image 
                labels.append(root.split("/")[-1])
    
    if forward_infer:
        # no labels in forward inference
        return image_paths
    
    # encode labels given encoder
    labels = label_encoder.transform(labels)
    return image_paths, labels


def get_image_metadata(image_paths, yaml_args, meta_stats_dicts = None):
    """
    Returns image metadata. 

    Input:
        image_paths (list) -> paths to images  
        yaml_args
        meta_stats_dicts (bool) -> None if not forward-inference. Otherwise, (dict) used to pull metadata columns in proper order given keys

    Output: 
        image_meta (list)

    Hard-coded hyperparameters: 
        None
    """
    image_col = yaml_args["meta_image_name_column"]
    # if metadata statistics exist (forward inference)
    if meta_stats_dicts is None:
        # use yaml args to get the proper columns 
        meta_cols = yaml_args["meta_columns"]
    else:
        # pull column names from the keys of the inputted dictionary 
        meta_cols = list(meta_stats_dicts["metadata_mean_dict"].keys())
    
    meta_data = pd.read_csv(yaml_args["metadata_csv"])
    metadata_array = meta_data[meta_cols].values
    # number of metadata observations 
    length_of_meta_obs = len(metadata_array[0])
    nan_counter = 0
    image_meta = []
    for idx, image in enumerate(image_paths):
        try:
            # get the image name with file ending (no file ending commented out)
            image_name = image.split('/')[-1]#.replace(".jpg", "")
            #image_name = image_name.replace(".png", "")
            #image_name = image_name.replace(".jpeg", "")
            # find the values in the csv 
            values = meta_data[meta_data[image_col] == image_name][meta_cols].values[0]
            image_meta.append(values.tolist())

        except IndexError:
            # metadata for image not found in csv 
            print(image_name, "not present in metadata file:", image)
            nan_counter += 1
            image_meta.append([np.nan] * length_of_meta_obs)

    print("Number of images missing metadata:", nan_counter)
    return image_meta



def metadata_impute_and_normalize(df, mean_array, std_array):
    """
    Replaces all nans in given df with mean values and uses mean and std arrays to normalize the data. 
    Alters "metadata" column of given df. 
    """
    metadata_array = [np.array(meta) for meta in df["metadata"]]
    for i, meta in enumerate(metadata_array):
        # identify nans 
        nan_indices = np.isnan(meta)
        # replace with mean 
        metadata_array[i][nan_indices] = mean_array[nan_indices]
        # standardize 
        norm_meta = (meta - mean_array) / std_array
        metadata_array[i] = norm_meta
    df["metadata"] = list(metadata_array)



def get_data_loader (yaml_args, transform, label_encoder = None, metadata_stats_dicts = None):
    """
    Creates a torch DataLoader given yaml file and transform. 
    label_encoder is not None in event of train. 
    meta_stats_dicts is not None in event of forward-inference.
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
            # no need for labels in forward-inference 
            full_data = pd.DataFrame({"images" : images})
            dt = MyDataset(yaml_args, full_data["images"], transform=transform)
        # no need for train-test-val split in this case either 
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

    # reset indices to avoid indexing issues 
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    test_df = test_df.reset_index()

    if (yaml_args["include_metadata"]): 
        print("Replacing nans and normalizing data.")
        train_metadata_array = np.array([np.array(meta) for meta in train_df["metadata"]])
        has_nan = np.isnan(train_metadata_array).any(axis=1)
        rows_with_nan = train_metadata_array[has_nan]
        print(len(rows_with_nan), "rows from train set have nan values. Ignoring these rows in mean and std calculation.")

        # calculate mean and std for given train set 
        train_metadata_mean = np.nanmean(train_metadata_array, axis=0)
        train_metadata_std = np.nanstd(train_metadata_array, axis=0)
        train_metadata_std[train_metadata_std == 0] = 1.0

        # impute and normalize train, val, and test given those values 
        metadata_impute_and_normalize(train_df, train_metadata_mean, train_metadata_std)
        metadata_impute_and_normalize(test_df, train_metadata_mean, train_metadata_std)
        metadata_impute_and_normalize(val_df, train_metadata_mean, train_metadata_std)

        train_dataset = MyDataset(yaml_args, train_df["images"], train_df["labels"], train_df["metadata"], transform)
        test_dataset = MyDataset(yaml_args, test_df["images"], test_df["labels"], test_df["metadata"], transform)
        val_dataset = MyDataset(yaml_args, val_df["images"], val_df["labels"], val_df["metadata"], transform)
        # zip mean and std dict with colnames to preserve order and not inadvertently shuffle data in the future 
        mean_dict = dict(zip(yaml_args["meta_columns"], train_metadata_mean))
        std_dict = dict(zip(yaml_args["meta_columns"], train_metadata_std))
    
    else:
        # no metadata, no imputation or normalization needed 
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