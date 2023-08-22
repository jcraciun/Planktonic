# Zooplankton-Pytorch (Julia's version)
Note: file copied from https://github.com/illinijeff/zooml.git

## Setup

First, install a Python3 version of Anaconda from here:
`https://www.anaconda.com/distribution/`

The following commands will install required libraries such as numpy, pandas, and 
PyTorch and create the appropriate conda environment named 'zooml':

`$ conda env create -f ./conda_environments/environment.yaml`\
`$ conda activate planktonic_env`

### Pulling Source code from Git

For the current private repo, hosted on github, the repo URL and command are as follows:

`git clone https://github.com/jcraciun/Planktonic.git`

This will prompt for username/password unless you have set up a public / private key 
pair with github and use the ssh version

## Training New Models
### Using folders of images as input

The dataset folder should have 1 level of subdirectories that are named by the class of
the images inside the subdirectory. Within these subdirectories, all the files should be image files, 
but alternative formats can be specified as parameters to the dataset in the code. 

NOTE: ENSURE ALL FILES IN DATA_DIR ARE FOLDERS FOR CLASSES. You will get errors/undefined behavior otherwise. 

Example dataset of images:

```
    <DATA_DIR>/
    ├── Acantharians
    ├── Copepods
    ├── Marine Snow
    └── Siphonophores
```

### Deploy Trial
After setting up your .yaml file, run the following command to deploy a trial:

`python3 -m main filename.yml`


### Directory Structure
After training at least one model, the following directory structure will be created to
be used for all future experiments. Note that all <CAPITALIZED> words are replaced with
their values from the configuration file. Briefly, all training sessions are considered
"trials" of an "experiment". Each experiment is named by its model type and its train
ID. Each trial will keep a cache of computed data such as sampling indices or image
mean and standard deviation for reprodicibility. These are also centrally cached in
"datasets/". You can also resume training a model by using its "args.yaml".
```
    Directory structure is
    <MAIN>/
    ├── datasets/
    │   └── <DATASET_NAME>/                        <-- Dataset metadata files (e.g. image statistics)
    └── <MODEL_TYPE>_<TRAIN_ID>/
        └── trial_<TRIAL_ID>/
            ├── checkpoints/
            │   └── best_model.pth.tar             <-- Best saved PyTorch model parameters
            ├── deployment_metadata/
            │   ├── dataset.csv                    <-- Dataset name, sampling indices, class names, size
            │   ├── image_statistics.csv           <-- Mean, std across all images from training set
            │   ├── image_metadata_statistics.csv  <-- Mean, std, max across all XMP metadata from training set
            │   └── training.csv                   <-- Early stopping count, epoch count
            ├── logs/
            │   ├── args.yaml                      <-- Saved config
            │   ├── train.log                      <-- Train log
            │   ├── train_loss.csv                 <-- Train losses
            │   └── val_loss.csv                   <-- Validation losses
            └── outputs/
                ├── graph_for_train_val_loss.png   <-- Graph of train versus validation loss
                ├── classifications_for_*.csv      <-- True and predicted class for each image
                ├── confusion_matrix_for_*.png     <-- Confusion matrix image for _ test
                └── confusion_matrix_for_*.csv     <-- Confusion matrix for _ test
```

## Results of Training

### Examining Confusion Matrix

### Understanding Training Log file

The log file contains records for each epoch of training, including execution time and current log loss.

### Model / Checkpoint Save structure

## Testing / Applying a trained model to unlabeled images

## Modifications / Updates

### Checking in code changes into Github

Note that the following sections assume that you are working on your own Github fork 
of the repository since you will not have write permissions otherwise

https://help.github.com/en/github/getting-started-with-github/fork-a-repo

If using github, your local environment may need a couple of updates. Specifically, github will check to see
not only that your username/password is correct, but also that your email address matches. And if you've opted into
Github's privacy features, that email is likely something like 
`##########+userid@users.noreply.github.com`

https://help.github.com/en/github/setting-up-and-managing-your-github-user-account/setting-your-commit-email-address

### Updating Conda Environment

If new libraries are added, the conda environment can be updated with:

`$ conda env update -f ./conda_environments/environment.yaml --prune`
