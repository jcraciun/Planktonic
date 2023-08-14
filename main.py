import torch 
from torch import nn
from tqdm import tqdm
import os 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from models.set_model import set_model
from load_plankton.create_dataloader import split_and_load
from load_plankton.utils import EarlyStopping
import yaml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import numpy as np
import shutil
import ast

def train(model, trainloader, criterion, optimizer, epoch, include_metadata, detect_anomaly = False):
    if save_model:
        if not isinstance(save_directory, str):
            raise Exception("save_model is set to True but no directory specified or directory not inputted as string.")
        if not os.path.exists(save_directory):
            print("Save directory does not exist. Creating a directory at inputted path.")
            os.makedirs(save_directory)
    model.train()
    torch.autograd.set_detect_anomaly(detect_anomaly)
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        image, labels, metadata_batch, _ = data
        if include_metadata:
            metadata_batch = metadata_batch.to('cuda')
            metadata_batch = metadata_batch.float()
        counter += 1
        image = image.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        # Forward pass.
        if include_metadata:
            outputs = model(image, metadata_batch)
        else:
            outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc, loss

def validate(model, testloader, criterion, optimizer, epoch, include_metadata):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels, metadata_batch, _ = data
            if include_metadata:
                metadata_batch = metadata_batch.to('cuda')
                metadata_batch = metadata_batch.float()
            counter += 1
            image = image.to('cuda')
            labels = labels.to('cuda')
            # Forward pass
            if include_metadata:
                outputs = model(image, metadata_batch)
            else:
                outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

def test(model, save_file_path, test_dataloader, classes, include_metadata=False):
    model.load_state_dict(torch.load(save_file_path))
    model.to("cuda")
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels, metadata_batch, _ in tqdm(test_dataloader):
            images = images.to("cuda")
            labels = labels.to("cuda")
            if include_metadata:
                metadata_batch = metadata_batch.to("cuda")
                metadata_batch = metadata_batch.float()
                outputs = model(images, metadata_batch)
            else:
                outputs = model(images)
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth
    

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    print(df_cm)
    target = ["Category {}".format(i) for i in range(len(classes))]
    print(classification_report(y_true, y_pred, target_names=classes))
    scores = np.diag(cf_matrix)/cf_matrix.sum(1)
    scores = pd.DataFrame(scores, index = classes, columns = ["scores"])
    print(scores)
    print("Balanced accuracy score:")
    print(balanced_accuracy_score(y_true, y_pred))

def forward_infer(model, save_file_path, image_directory_path, dataloader, classes, include_metadata=False):
    print("Loading model for forward inference.")
    model.load_state_dict(torch.load(save_file_path))
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        print("Applying predicted labels to images.")
        for images, labels, metadata_batch, paths in tqdm(dataloader):
            images = images.to("cuda")
            labels = labels.to("cuda")
            if include_metadata:
                metadata_batch = metadata_batch.to("cuda")
                metadata_batch = metadata_batch.float()
                outputs = model(images, metadata_batch)
            else:
                outputs = model(images)
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            print("Placing images in predicted class directories.")
            for i in range(len(output)):
                source_file = paths[i]
                destination_directory = image_directory_path + '/' + classes[output[i]]
                if (os.path.exists(destination_directory)):
                    shutil.copy(source_file, destination_directory)
                else:
                    print("Directory not found. Creating one for:", classes[output[i]])
                    os.makedirs(destination_directory)
                    shutil.copy(source_file, destination_directory)

def main():
    print("Parsing config.yml.")
    with open('output.yaml', 'r') as file:
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

    print("Selected", yaml_args["comb_method"], "method with parameters", yaml_args["comb_config"])
    print("Setting model and transforms.")
    model, img_transforms = set_model(model_name = yaml_args["model"], num_class = yaml_args["num_classes"], p_dropout = yaml_args["p_dropout"], 
                                                comb_method = yaml_args["comb_method"], comb_config = comb_config, neurons_reducer_block = yaml_args["neurons_reducer_block"])
    
    classes = os.listdir(yaml_args["path_to_image_folder"])
    print("Creating dataloaders.")

    # forward inference segment (doesn't create train-test-split)
    if yaml_args["run_type"] == "forward_infer":
        dataloader = split_and_load(yaml_args, img_transforms)
        forward_infer(model, yaml_args["path_to_load_model"], yaml_args["path_to_copy_infer_images"], dataloader, classes, include_metadata=yaml_args["include_metadata"])
    else:
        train_dataloader, test_dataloader, val_dataloader = split_and_load(yaml_args, img_transforms)

    if yaml_args["run_type"] == "train":
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=yaml_args["adam_learning_rate"])

        if yaml_args["early_stopper"]:
            early_stopper = EarlyStopping(patience=yaml_args["es_patience"], delta = yaml_args["es_min_delta"])

        model.to('cuda')
        num_epochs = yaml_args["num_epochs"]

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        for epoch in range(num_epochs):
            if yaml_args["early_stopper"] and early_stopper.should_stop():
                print(
                    f"Validation has not improved over {early_stopper.count}"
                    f" epochs (including previous runs). Early stopping..."
                )
                break
            # third loss argument for model-saving purposes
            train_epoch_loss, train_epoch_acc, loss = train(model, train_dataloader, criterion, optimizer, 
                                                    epoch, yaml_args["include_metadata"])
            valid_epoch_loss, valid_epoch_acc = validate(model, val_dataloader, criterion, optimizer, epoch, yaml_args["include_metadata"])
        if yaml_args["save_model"]:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'losslogger': loss.item(), }
        # save model
            if yaml_args["save_only_best_model"]:
                if valid_epoch_loss > max(valid_loss):
                    print(epoch+1, "is the new best model. Saving to file.")
                    torch.save(model.state_dict(), os.path.join(yaml_args["path_to_save_epochs"], 'epoch-{}.pth'.format(epoch+1)))
            else:
                torch.save(model.state_dict(), os.path.join(yaml_args["path_to_save_epochs"], 'epoch-{}.pth'.format(epoch+1)))
    
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)
        
            print("Epoch: ", epoch)    
            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
            print('-'*50)

    elif yaml_args["run_type"] == "test":
        print("Testing")
        test(model, yaml_args["path_to_load_model"], test_dataloader, classes, include_metadata=yaml_args["include_metadata"])

main() 

