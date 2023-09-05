import torch 
from torch import nn
from tqdm import tqdm
import os 
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sn



def train(model, trainloader, criterion, optimizer, include_metadata, detect_anomaly = False):
    """
    Trains inputted model and returns epoch_loss, epoch_acc.

    Input:
        model 
        trainloader (custom Torch DataLoader)
        criterion 
        optimizer
        include_metadata (bool)

    Output: 
        epoch_loss (int),
        epoch_acc (int)

    Hard-coded hyperparameters: 
        detect_anomaly (see function declaration) -> set to True to turn Torch diagnostics, slows training time significantly.

    """

    model.train()
    torch.autograd.set_detect_anomaly(detect_anomaly)
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        image, labels, metadata_batch, _ = data
        # metadata inclusion 
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
    return epoch_loss, epoch_acc

def validate(model, validloader, criterion, include_metadata):
    """
    Validates training and returns epoch_loss, epoch_acc.

    Input:
        model 
        validloader (custom Torch DataLoader)
        criterion 
        include_metadata (bool)

    Output: 
        epoch_loss (int),
        epoch_acc (int)

    Hard-coded hyperparameters: 
        None

    """
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(validloader), total=len(validloader)):
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
    epoch_acc = 100. * (valid_running_correct / len(validloader.dataset))
    return epoch_loss, epoch_acc

def test(model, save_file_path, testloader, classes, save_plot_path, include_metadata=False):
    """
    Tests given model from save_file_path. Saves confusion matrix, Sklearn classification report, and balanced accuracy score to proper folders. 

    Input:
        model 
        save_file_path -> path towards desired model 
        testloader (custom Torch DataLoader)
        classes (list) -> list of strings that include the classes in label order
        save_plot_path -> path to .png of results 
        include_metadata (bool)

    Output: 
        None

    Hard-coded hyperparameters: 
        None

    """
    model.load_state_dict(torch.load(save_file_path)['state_dict'])
    model.to("cuda")
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels, metadata_batch, _ in tqdm(testloader):
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
    
   
    print("If cf_matrix produces a shape error, ensure that all the files in the image directory pertain to a class within the dataloader.")

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                            columns=[i for i in classes])
    print(df_cm)
        
    target = ["Category {}".format(i) for i in range(len(classes))]
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
        
    scores = np.diag(cf_matrix) / cf_matrix.sum(1)
    scores = pd.DataFrame(scores, index=classes, columns=["scores"])
    print(scores)
        
    print("Balanced accuracy score:")
    print(balanced_accuracy_score(y_true, y_pred))

    print("Saving figures to results directory.")
    # Create plots from printed data
    plt.figure(figsize=(20, 16))

    # Confusion Matrix
    #plt.subplot(231)
    plt.title("Confusion Matrix")
    sn.heatmap(df_cm, annot=True)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    #plt.tight_layout()
    confusion_matrix_filename = 'confusion_matrix.png'
    confusion_matrix_filepath = os.path.join(save_plot_path, confusion_matrix_filename)
    plt.savefig(confusion_matrix_filepath)
    plt.clf()

    plt.figure(figsize=(20, 16))
    # Classification Report
    plt.subplot(232)
    plt.title("Classification Report")
    plt.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')
    plt.tight_layout()

    plt.subplot(233)  # Adjust the subplot arrangement
    plt.title("Scores")
    plt.table(cellText=scores.values, colLabels=['Score'], rowLabels=scores.index, loc='center')
    plt.axis('off')  # Turn off the axis

    # Balanced Accuracy Score
    plt.subplot(212)  # Adjust the subplot arrangement
    plt.text(0.5, 0.5, f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.2f}",
            horizontalalignment='center', verticalalignment='center', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    output_filename = 'results.png'
    output_filepath = os.path.join(save_plot_path, output_filename)
    #plt.tight_layout()
    plt.savefig(output_filepath)
    

def forward_infer(model, image_directory_path, dataloader, classes, include_metadata=False):
    """
    Classifies images and sorts them into respective folders. Creates a folder for class if nonexistent. 

    Input:
        model 
        image_directory_path -> path to sort images  
        dataloader (custom Torch DataLoader)
        classes (list) -> list of strings that include the classes in label order
        include_metadata (bool)

    Output: 
        None

    Hard-coded hyperparameters: 
        None

    """
    print("Loading model for forward inference.")
    model.to("cuda")
    model.eval()
    with torch.no_grad():
        print("Applying predicted labels to images.")
        for images, _, metadata_batch, paths in tqdm(dataloader):
            images = images.to("cuda")
            if include_metadata:
                metadata_batch = metadata_batch.to("cuda")
                metadata_batch = metadata_batch.float()
                outputs = model(images, metadata_batch)
            else:
                outputs = model(images)
            output = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            # copy the images into the proper folder 
            for i in range(len(output)):
                source_file = paths[i]
                destination_directory = image_directory_path + '/' + classes[output[i]]
                # check if directory exists 
                if (os.path.exists(destination_directory)):
                    shutil.copy(source_file, destination_directory)
                else:
                    # create new one otherwise then copy image over
                    print("Directory not found. Creating one for:", classes[output[i]])
                    os.makedirs(destination_directory)
                    shutil.copy(source_file, destination_directory)
    print("All images sorted into directories.")