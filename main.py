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

    # TO DO: SAVE YAML 
    # list of classes 
    classes = os.listdir(yaml_args["path_to_image_folder"])
    le = LabelEncoder()
    le.fit(classes)
    num_classes = len(classes)
    print("Number of classes:", num_classes)
    print("Classes:", classes)
    print("Selected", yaml_args["comb_method"], "method with parameters", yaml_args["comb_config"])
    print("Setting model and transforms.")

    
    # model and transforms 
    if yaml_args["run_type"] == "train":
        model, img_transforms = set_model(model_name = yaml_args["model"], num_class = num_classes, p_dropout = yaml_args["p_dropout"], 
                                                comb_method = yaml_args["comb_method"], comb_config = comb_config, neurons_reducer_block = yaml_args["neurons_reducer_block"])
    else:
        print("Loading saved model parameters.")
        print("Loading checkpoint dictionary:", checkpoint['init_args'])
        # TO DO: LOADING METADATA COLUMNS -> INCLUDE METADATA FLAG ETC
        checkpoint = torch.load(yaml_args["path_to_load_model"])
        model, img_transforms = set_model(**checkpoint['init_args']) 
        model.load_state_dict(torch.load(yaml_args["path_to_load_model"])['state_dict'])

    print("Creating dataloaders.")
    # forward inference segment (doesn't create train-test-split)
    if yaml_args["run_type"] == "forward_infer":
        dataloader = split_and_load(yaml_args, img_transforms, le)
        forward_infer(model, yaml_args["path_to_copy_infer_images"], dataloader, classes, include_metadata=yaml_args["include_metadata"])
    else:
        # train-test-split
        train_dataloader, test_dataloader, val_dataloader = split_and_load(yaml_args, img_transforms, le)

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
                     "init_args" : {
                            'model_name': yaml_args["model"],
                            'num_class': num_classes,
                            'p_dropout': yaml_args["p_dropout"],
                            'comb_method': yaml_args["comb_method"],
                            'comb_config': comb_config,
                            'neurons_reducer_block': yaml_args["neurons_reducer_block"]
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
                
            if yaml_args["include_metadata"]:
                if yaml_args["comb_method"] == "metablock":
                    comb_config = (yaml_args["comb_config"], len(yaml_args["meta_columns"]))
                elif yaml_args["comb_method"] == "metanet":
                    config = ast.literal_eval(yaml_args["comb_config"])
                    comb_config = (len(yaml_args["meta_columns"]), config[0], config[1])
                elif yaml_args["comb_method"] == "concat":
                    comb_config = (len(yaml_args["meta_columns"]))
            else:
                print("Selected to not include metadata. Changing any comb_config and comb_method input to None.")
                comb_config = None
                yaml_args["comb_method"] = None
                yaml_args["comb_config"] = None
                            
            if yaml_args["run_type"] == "train":
                path = os.path.join(os.getcwd(), "trials", yaml_args["model"] + "_" + str(yaml_args["comb_method"]) + "_" + str(yaml_args["trial_id"]))
             # avoid overwriting existing trials 
                if (os.path.exists(path)):
                    raise Exception("The directory to save this model already exists. Give the .yml file a new trial_id and/or model to avoid overwriting old results.")
                else:
                    print("Making directories at:", path)
                    os.makedirs(path)
                    os.makedirs(os.path.join(path, "checkpoints"))
                    os.makedirs(os.path.join(path, "logs"))
                    os.makedirs(os.path.join(path, "results"))
                    results_directory = os.path.join(path, "results")
                    path_to_save_epochs = os.path.join(path, "checkpoints")
                    shutil.copy(filename, os.path.join(path, "logs", "args.yml"))
                    logs_directory = os.path.join(path, "logs")
                    shutil.move(output_file_name, os.path.join(logs_directory, output_file_name))

            main()
