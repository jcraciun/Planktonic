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


def main():
    """
    Sets .yml configs and calls functions from run_types.py as needed. 
    Hard-coded hyperparameters: 
        optimizer -> CrossEntropyLoss()
        loss -> torch.optim.Adam()

    """

    # resume training 
    if filename == "args.yml":
        yaml_args["run_type"] = "resume_training"
        path = yaml_args["trial_path"]
        epoch_path = os.path.join(path, "checkpoints")
        epoch_files = [file for file in os.listdir(epoch_path) if file.startswith("epoch-")]
        # find last saved epoch 
        if epoch_files:
            last_epoch_file = max(epoch_files, key=lambda x: int(x.split("-")[1].split(".")[0]))
            print(f"The filename of the last epoch is: {last_epoch_file}")
            print("Loading saved model parameters.")
            checkpoint = torch.load(os.path.join(epoch_path, last_epoch_file))
            # if the final logged epoch is the same as the max_epoch of the trial
            if checkpoint['epoch'] == yaml_args["num_epochs"]:
                raise Exception("Trial reached max epochs specified in .yml file. Start a new trial.")
            # load the model dictionary 
            print("Loading checkpoint dictionary:", checkpoint['init_args'])
            classes = checkpoint["label_encoder"].classes_
            num_classes = len(classes)
            le = checkpoint["label_encoder"]
            comb_config = checkpoint["init_args"]["comb_config"]
            image_means = checkpoint["init_args"]["transforms_mean"]
            image_stds = checkpoint["init_args"]["transforms_std"]
            print("Setting model and transforms.")
            model, img_transforms = set_model(**checkpoint['init_args']) 
            # dump new arguments in a .yaml file 
            with open(filename, "w") as yaml_file:
                yaml.dump(yaml_args, yaml_file)
            # create new files to avoid overriting initial train session 
            shutil.copy(filename, os.path.join(path, "logs", "resume_train_args.yml"))
            shutil.move(output_file_name, os.path.join(path, "logs", "resume_train_output.txt"))

        else:
            # args.yml doesn't correspond to folder in directory 
            raise Exception("No epoch files found in the folder for the current args.yml.")


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

        forward_infer(model, yaml_args["path_to_copy_infer_images"], dataloader, classes, include_metadata=yaml_args["include_metadata"])
        return 0

    if yaml_args["run_type"] == "train":
        if yaml_args["include_metadata"]:
            # set proper comb_methods 
            if yaml_args["comb_method"] == "metablock":
                config = ast.literal_eval(yaml_args["comb_config"])
                comb_config = (config[0], len(yaml_args["meta_columns"]), config[1]) 
            elif yaml_args["comb_method"] == "metanet":
                config = ast.literal_eval(yaml_args["comb_config"])
                comb_config = (len(yaml_args["meta_columns"]), config[0], config[1], config[2], config[3])
            elif yaml_args["comb_method"] == "concat":
                comb_config = (len(yaml_args["meta_columns"]))
        else:
            # no metadata, clear metadata-related arguments 
            print("Selected to not include metadata. Changing any comb_config and comb_method input to None.")
            comb_config = None
            yaml_args["comb_method"] = None
            yaml_args["comb_config"] = None
                        
      # create folders for trial 
        path = os.path.join(os.getcwd(), "trials", yaml_args["model"] + "_" + str(yaml_args["comb_method"]) + "_" + str(yaml_args["trial_id"]))
        yaml_args["trial_path"] = path
        # warn if overriding existing trials 
        if (os.path.exists(path)):
            print("The directory to save this model already exists. Overwriting old results at:", path)
        else:
            print("Making directories at:", path)
        
        # dump cleaned arguments into yaml file for resume-training purposes 
        with open(filename, "w") as yaml_file:
            yaml.dump(yaml_args, yaml_file)

        # make relevant directories 
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(path, "results"), exist_ok=True)
        logs_directory = os.path.join(path, "logs")
        # copy the updated yaml file over 
        shutil.copy(filename, os.path.join(path, "logs", "args.yml"))
        # move the output file into the proper folder 
        shutil.move(output_file_name, os.path.join(logs_directory, output_file_name))

        # model and transforms 
        classes = os.listdir(yaml_args["path_to_image_folder"])
        le = LabelEncoder()
        le.fit(classes)
        num_classes = len(classes)
        print("Number of classes:", num_classes)
        print("Classes:", classes)

        # image statistics 
        if yaml_args["image_norm_csv"] == 'None':
            image_means = None
            image_stds = None
        else:
            image_data = pd.read_csv(yaml_args["image_norm_csv"])
            image_means = list(image_data.means)
            image_stds = list(image_data.stds)


        print("Setting model and transforms.")
        model, img_transforms = set_model(model_name = yaml_args["model"], num_class = num_classes, p_dropout = yaml_args["p_dropout"], 
                                                comb_method = yaml_args["comb_method"], comb_config = comb_config, neurons_reducer_block = yaml_args["neurons_reducer_block"],
                                                transforms_mean = image_means, transforms_std = image_stds)
        print("Selected", yaml_args["comb_method"], "method with parameters", comb_config)

        start_epoch = 0

    results_directory = os.path.join(path, "results")
    path_to_save_epochs = os.path.join(path, "checkpoints")
    logs_directory = os.path.join(path, "logs")
    # create txt files to dump losses and accuracies 
    train_loss_file = os.path.join(logs_directory, "train_losses.txt")
    valid_loss_file = os.path.join(logs_directory, "valid_losses.txt")
    train_acc_file = os.path.join(logs_directory, "train_acc.txt")
    valid_acc_file = os.path.join(logs_directory, "valid_acc.txt")


    print("Creating dataloaders.")
    if yaml_args["include_metadata"]:
        train_dataloader, test_dataloader, val_dataloader,  meta_mean_dict, meta_std_dict,  = split_and_load(yaml_args, img_transforms, le)
    else: 
        train_dataloader, test_dataloader, val_dataloader = split_and_load(yaml_args, img_transforms, le)
        meta_mean_dict = None
        meta_std_dict = None

    model.to('cuda')
    # train 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=yaml_args["adam_learning_rate"])
    num_epochs = yaml_args["num_epochs"]

    if yaml_args["run_type"] == "resume_training":
        # load old optimizer and epoch starting point 
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    if yaml_args["early_stopper"]:
        early_stopper = EarlyStopping(patience=yaml_args["es_patience"], delta = yaml_args["es_min_delta"])

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
        # write new losses and accuracies into txt file 
        with open(train_loss_file, "a") as train_loss_f:
            train_loss_f.write(f"Epoch {epoch + 1}: Training Loss: {train_epoch_loss:.3f}\n")
        with open(valid_loss_file, "a") as valid_loss_f:
            valid_loss_f.write(f"Epoch {epoch + 1}: Validation Loss: {valid_epoch_loss:.3f}\n")
        with open(train_acc_file, "a") as train_acc_f:
            train_acc_f.write(f"Epoch {epoch + 1}: Training Loss: {train_epoch_acc:.3f}\n")
        with open(valid_acc_file, "a") as valid_acc_f:
            valid_acc_f.write(f"Epoch {epoch + 1}: Validation Loss: {valid_epoch_acc:.3f}\n")

        print("Epoch: ", epoch + 1)    
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

    # test 
    # select last saved model 
    best_epoch_path = os.path.join(path_to_save_epochs, 'epoch-{}.pth'.format(best_epoch))
    print("Selected model for testing from:", best_epoch_path)
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    # open loss and accuracy files to plot data 
    with open(train_loss_file, "r") as train_loss_f:
        train_losses = [float(line.split(":")[-1]) for line in train_loss_f.readlines()]
    with open(valid_loss_file, "r") as valid_loss_f:
        valid_losses = [float(line.split(":")[-1]) for line in valid_loss_f.readlines()]
    with open(train_acc_file, "r") as train_acc_f:
        train_accs = [float(line.split(":")[-1]) for line in train_acc_f.readlines()]
    with open(valid_acc_file, "r") as valid_acc_f:
        valid_accs = [float(line.split(":")[-1]) for line in valid_acc_f.readlines()]

    save_acc_loss_plots(train_losses, valid_losses, train_accs, valid_accs, logs_directory)
    # load checkpoint and image transforms 
    checkpoint = torch.load(best_epoch_path)
    model, img_transforms = set_model(**checkpoint['init_args']) 
    test(model, best_epoch_path, test_dataloader, le.classes_, results_directory, yaml_args["include_metadata"])
               

class Tee:
    # class to allow all printed output to be written to an output.txt file 
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
