import torch 
from torch import nn
from tqdm import tqdm
import os 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from models.set_model import set_model
from load_plankton.create_dataloader import split_and_load
from load_plankton.utils import EarlyStopping, save_acc_loss_plots
import yaml
import numpy as np
import numpy as np
import shutil
import ast
import sys
from run_types import train, test, forward_infer, validate
import contextlib


def main():
    """
    Hard-coded hyperparameters: 
        optimizer -> CrossEntropyLoss()
        loss -> torch.optim.Adam()

    """

    if yaml_args["run_type"] == "forward_infer":
        print("Loading saved model parameters.")
        checkpoint = torch.load(yaml_args["path_to_load_model"])
        print("Loading checkpoint dictionary:", checkpoint['init_args'])
        classes = checkpoint["label_encoder"].classes_
        print("Classes used for inference:", classes)
        # model has no metadata, but present in yaml
        if checkpoint["init_args"]["comb_method"] is None and yaml_args["include_metadata"]:
            raise Exception("The model you loaded does not allow for metadata. Either set include_metadata to False or load an appropriate model.")
        # model has metadata, but missing in yaml
        if checkpoint["init_args"]["comb_method"] is not None and not yaml_args["include_metadata"]:
            raise Exception("The model you loaded expects metadata. Either set include_metadata to True or load an appropriate model.")
        model, img_transforms = set_model(**checkpoint['init_args']) 
        model.load_state_dict(torch.load(yaml_args["path_to_load_model"])['state_dict'])
        # check to ensure all metadata columns from saved model are present
        if yaml_args["include_metadata"]:
            metadata_dicts = checkpoint["transfer_args"]
            meta_csv = pd.read_csv(yaml_args["metadata_csv"])
            missing_cols = []
            for key in metadata_dicts["metadata_mean_dict"]:
                if not key in set(meta_csv.columns):
                    missing_cols.append(key)

            if (len(missing_cols) > 0):
                raise Exception("The csv you loaded is missing the following metadata columns the model was trained on:", missing_cols)
            else:
                print("All metadata columns used in training are present.")
                dataloader = split_and_load(yaml_args, img_transforms, metadata_stats_dicts=  metadata_dicts)
        else:
            dataloader = split_and_load(yaml_args, img_transforms)
        #model, img_transforms = set_model(**checkpoint['init_args']) 
        #model.load_state_dict(torch.load(yaml_args["path_to_load_model"])['state_dict'])
        #dataloader = split_and_load(yaml_args, img_transforms, checkpoint["label_encoder"], metadata_dicts)
        forward_infer(model, yaml_args["path_to_copy_infer_images"], dataloader, classes, include_metadata=yaml_args["include_metadata"])
        return 0

    if yaml_args["run_type"] == "train":
        # sort comb config and clean up params from yaml 
        if yaml_args["include_metadata"]:
            if yaml_args["comb_method"] == "metablock":
                config = ast.literal_eval(yaml_args["comb_config"])
                comb_config = (config[0], len(yaml_args["meta_columns"]), config[1]) 
            elif yaml_args["comb_method"] == "metanet":
                config = ast.literal_eval(yaml_args["comb_config"])
                comb_config = (len(yaml_args["meta_columns"]), config[0], config[1], config[2], config[3])
            elif yaml_args["comb_method"] == "concat":
                comb_config = (len(yaml_args["meta_columns"]))
        else:
            print("Selected to not include metadata. Changing any comb_config and comb_method input to None.")
            comb_config = None
            yaml_args["comb_method"] = None
            yaml_args["comb_config"] = None
                        
      # create folders for trial 
        path = os.path.join(os.getcwd(), "trials", yaml_args["model"] + "_" + str(yaml_args["comb_method"]) + "_" + str(yaml_args["trial_id"]))
        # avoid overwriting existing trials 
        #if (os.path.exists(path)):
        #    raise Exception("The directory to save this model already exists. Give the .yml file a new trial_id and/or model to avoid overwriting old results.")
        #else:
        
        print("Making directories at:", path)
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(path, "results"), exist_ok=True)
        results_directory = os.path.join(path, "results")
        path_to_save_epochs = os.path.join(path, "checkpoints")
        shutil.copy(filename, os.path.join(path, "logs", "args.yml"))
        logs_directory = os.path.join(path, "logs")
        shutil.move(output_file_name, os.path.join(logs_directory, output_file_name))


    # model and transforms 
        classes = os.listdir(yaml_args["path_to_image_folder"])
        le = LabelEncoder()
        le.fit(classes)
        num_classes = len(classes)
        print("Number of classes:", num_classes)
        print("Classes:", classes)

        if yaml_args["image_norm_csv"] == 'None':
            image_means = None
            image_stds = None
        else:
            image_data = pd.read_csv(yaml_args["image_norm_csv"])
            image_means = list(image_data.means)
            image_means = list(image_data.stds)


        print("Setting model and transforms.")
        model, img_transforms = set_model(model_name = yaml_args["model"], num_class = num_classes, p_dropout = yaml_args["p_dropout"], 
                                                comb_method = yaml_args["comb_method"], comb_config = comb_config, neurons_reducer_block = yaml_args["neurons_reducer_block"],
                                                transforms_mean = image_means, transforms_std = image_stds)
        print("Selected", yaml_args["comb_method"], "method with parameters", yaml_args["comb_config"])


    print("Creating dataloaders.")
    if yaml_args["include_metadata"]:
        train_dataloader, test_dataloader, val_dataloader,  meta_mean_dict, meta_std_dict,  = split_and_load(yaml_args, img_transforms, le)
    else: 
        train_dataloader, test_dataloader, val_dataloader = split_and_load(yaml_args, img_transforms, le)
        meta_mean_dict = None
        meta_std_dict = None

    # train 
    if yaml_args["run_type"] == "train" or yaml_args["run_type"] == "resume_training":
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=yaml_args["adam_learning_rate"])
        start_epoch = 0

        # TO DO: ENSURE RESUME_TRAINING PROPERLY CONTINUES MODEL 
        if yaml_args["run_type"] == "resume_training":
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']

        if yaml_args["early_stopper"]:
            early_stopper = EarlyStopping(patience=yaml_args["es_patience"], delta = yaml_args["es_min_delta"])

        model.to('cuda')
        num_epochs = yaml_args["num_epochs"]

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        for epoch in range(start_epoch, num_epochs):
            # early stopper activated 
            if yaml_args["early_stopper"] and early_stopper.should_stop():
                print(
                    f"Validation has not improved over {early_stopper.count}"
                    f" epochs (including previous runs). Early stopping..."
                )
                break

            # train 
            train_epoch_loss, train_epoch_acc = train(model, train_dataloader, criterion, optimizer, 
                                                    yaml_args["include_metadata"])
            valid_epoch_loss, valid_epoch_acc = validate(model, val_dataloader, criterion, yaml_args["include_metadata"])
            # results 
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
            print("Epoch: ", epoch)    
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)
            
            # update early stopper 
            if yaml_args["early_stopper"]:
                early_stopper.step(valid_epoch_loss)

            # save model 
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     "label_encoder" : le,
                     "init_args" : {
                            'model_name': yaml_args["model"],
                            'num_class': num_classes,
                            'p_dropout': yaml_args["p_dropout"],
                            'comb_method': yaml_args["comb_method"],
                            'comb_config': comb_config,
                            'neurons_reducer_block': yaml_args["neurons_reducer_block"],
                            'transforms_mean' : image_means, 
                            'transforms_std' : image_stds
                        },
                     "transfer_args" : {
                         "metadata_std_dict" : meta_std_dict,
                         "metadata_mean_dict" : meta_mean_dict 
                     }}
                # save only best 
            if valid_epoch_loss <= min(valid_loss):
                best_epoch = epoch + 1
                print("Epoch", epoch+1, "is the new best model. Saving to file.")
                torch.save(state, os.path.join(path_to_save_epochs, 'epoch-{}.pth'.format(epoch+1)))

        best_epoch_path = os.path.join(path_to_save_epochs, 'epoch-{}.pth'.format(best_epoch))
        print("Selected model for testing from:", best_epoch_path)
        save_acc_loss_plots(epoch, train_loss, valid_loss, train_acc, valid_acc, logs_directory)
        checkpoint = torch.load(best_epoch_path)
        model, img_transforms = set_model(**checkpoint['init_args']) 
        test(model, best_epoch_path, test_dataloader, le.classes_, results_directory, yaml_args["include_metadata"])
               

class Tee:
    def __init__(self, name, mode='w'):
        self.file = open(name, mode)
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


output_file_name = "output.txt"
# Call your main function
if __name__ == "__main__":
    with open(output_file_name, "w") as output_file:
        # Redirect stdout to both the terminal and the output file
        with Tee(output_file_name):
            # Get the filename from command-line arguments
            filename = sys.argv[1]

            print("Parsing", filename)
            with open(filename, 'r') as file:
                yaml_args = yaml.safe_load(file)

            main()
