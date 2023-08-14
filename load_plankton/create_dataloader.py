import torch
import torchvision.transforms as transforms
from sklearn import preprocessing
from PIL import Image
import os 
from torch.utils import data
import numpy as np
import pandas as pd

class MyDataset (data.Dataset):
    """
    This is the standard way to implement a dataset pipeline in PyTorch. We need to extend the torch.utils.data.Dataset
    class and implement the following methods: __len__, __getitem__ and the constructor __init__
    """

    def __init__(self, yaml_args, transform=None):
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
        self.image_paths, self.labels = self._get_image_paths_labels(yaml_args["path_to_image_folder"])
        if yaml_args["include_metadata"]:
            self.image_paths, self.labels, self.meta_data = self._get_image_metadata(self.image_paths, self.labels, yaml_args)
        else:
            self.meta_data = None
        # if transform is None, we need to ensure that the PIL image will be transformed to tensor, otherwise we'll get an exception
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.ToTensor()

    def _get_image_paths_labels(self, base_path):
            image_paths = []
            labels = []
            for root, _, files in os.walk(base_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
                        labels.append(root.split("/")[-1])
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            labels = le.transform(labels)
            return image_paths, labels
    
    def _get_image_metadata(self, image_paths, labels, yaml_args):
            image_col = yaml_args["meta_image_name_column"]
            meta_cols = yaml_args["meta_columns"]
            meta_data = pd.read_csv(yaml_args["metadata_csv"])
            metadata_array = meta_data[meta_cols].values
            has_nan = np.isnan(metadata_array).any(axis=1)
            # Use boolean indexing to filter rows with NaN values
            rows_with_nan = meta_data[has_nan]
            print("Dropping", len(rows_with_nan), "rows with nan values.")
            meta_data.drop(rows_with_nan.index, inplace=True)
            metadata_array = meta_data[meta_cols].values
            # Calculate the mean and standard deviation along the desired axis (axis=0 for mean, axis=0 for standard deviation)
            metadata_mean = np.mean(metadata_array, axis=0)
            metadata_std = np.std(metadata_array, axis=0)
            metadata_std[metadata_std == 0] = 1.0
            # normalize values 
            norm_meta = (metadata_array - metadata_mean) / metadata_std
            # replace "meta" with normalized values 
            meta_data["meta"] = list(norm_meta)
            nan_counter = 0
            image_meta = []
            image_path_subset = []
            image_labels = []
            for idx, image in enumerate(image_paths):
                try:
                    image_name = image.split('/')[-1].replace(".png", "")
                    values = meta_data[meta_data[image_col] == image_name]["meta"].values[0]
                    image_meta.append(values.tolist())
                    image_path_subset.append(image)
                    image_labels.append(labels[idx])

                except IndexError:
                    print("Skipping image", image_name, " file path:", image)
                    nan_counter += 1

            print("Number of skipped images (missing metadata):", nan_counter)

            return image_path_subset, image_labels, image_meta
    
    def __len__(self):
        """ This method just returns the dataset size """
        return len(self.labels)


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
            #meta_data = self.meta_data[item]
            #print(self.meta_data[item])
            meta_data = self.meta_data[item]

        if self.labels is None:
            labels = []
        else:
            labels = self.labels[item]

        return image, labels, meta_data, img_id


def split_and_load(yaml_args, transforms):
    if yaml_args["run_type"] == "forward_infer":
        return get_data_loader(yaml_args, transforms)
    print("Splitting into train, test, and validation.")
    if yaml_args["include_metadata"]:
         train_dataloader, test_dataloader, val_dataloader = get_data_loader(yaml_args, transforms)
    else:
        train_dataloader, test_dataloader, val_dataloader = get_data_loader(yaml_args, transforms)

    return train_dataloader, test_dataloader, val_dataloader


def get_data_loader (yaml_args, transform, num_workers=4, pin_memory=True):
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
    
    dt = MyDataset(yaml_args, transform)

    if yaml_args["include_metadata"]:
        collate = custom_collate_fn
    else:
        collate = None 
    
    if yaml_args["run_type"] == "forward_infer":
        return data.DataLoader(dataset=dt, batch_size=yaml_args["batch_size"], shuffle=False, num_workers=num_workers,
                               pin_memory=pin_memory, collate_fn=collate)
    # train test split 
    train_size = int(yaml_args["train_proportion"] * len(dt))
    test_size = len(dt) - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dt, [train_size, test_size])
    torch.manual_seed(42)
    test_size = int(.5 * len(test_dataset))
    val_size = (len(test_dataset) - test_size)
    test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_size, val_size])
    print("Length of train dataset:", len(train_dataset))
    print("Length of validation dataset:", len(val_dataset))
    print("Length of test dataset:", len(test_dataset))

    train = data.DataLoader(dataset=train_dataset, batch_size=yaml_args["batch_size"], shuffle=True, num_workers=num_workers,
                            pin_memory=pin_memory, collate_fn=collate)
    test = data.DataLoader(dataset=test_dataset, batch_size=yaml_args["batch_size"], shuffle=False, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=collate)
    val = data.DataLoader(dataset=val_dataset, batch_size=yaml_args["batch_size"], shuffle=False, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=collate)

    return train, test, val

def custom_collate_fn(batch):
    """
    Ensures that the metadata is properly stacked and in order 
    (in some cases, data can become vertically split across rows instead of horizontally otherwise)
    """
    images, labels, meta_data, img_ids = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    meta_data_padded = torch.stack([torch.DoubleTensor(meta) for meta in meta_data])
    return images, labels, meta_data_padded, img_ids