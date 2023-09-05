# Zooplankton-PyTorch (Julia's version)
Note: file format and certain text copied from https://github.com/illinijeff/zooml.git

## Purpose and Signficance 

One of the Navy’s primary roles is to serve as an environmental steward as well as furthering the improvement of deep learning frameworks for underwater applications. Whitmore et al. [1] describes a correlation between certain environmental observations and Uncrewed Underwater Vehicle (UUV) sensor data quality degradation. Their research suggests that knowledge about environmental conditions could be leveraged to enhance image classification performance. 

Additionally, the Scripps Institution of Oceanography has a vested interest in the automation of plankton species classification to accurately assess the state of the ocean. As the foundation of the marine food chain, Plankton serve as a crucial signifier of the stability and condition of oceanic life. With NIWIC’s access to 1.2 million, 145 thousand, and 2.1 million labeled images from three types of cameras, Zooglider, UVP, and Zooscan respectively, plankton imagery provides a clear and interpretable testbed for metadata inclusion research while serving a valuable purpose in both improving Naval UUV sensor measurements and furthering environmental preservation efforts. 

*[1] B. M. Whitmore, J. S. Ellen and M. C. Grier, "Toward improving unmanned underwater vehicle sensing operations through characterization of the impacts and limitations of in situ environmental conditions," OCEANS 2022, Hampton Roads, Hampton Roads, VA, USA, 2022, pp. 1-4, doi: 10.1109/OCEANS47191.2022.9977345.*

## Setup

First, install a Python3 version of Anaconda from here:
`https://www.anaconda.com/distribution/`

The following commands will install required libraries such as numpy, pandas, and 
PyTorch and create the appropriate conda environment named 'planktonic_env':

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

**ENSURE ALL FILES IN DATA_DIR ARE FOLDERS FOR CLASSES. You will get errors/undefined behavior otherwise.**

Example dataset of images:

```
    <DATA_DIR>/
    ├── Acantharians
    ├── Copepods
    ├── Marine Snow
    └── Siphonophores
```

### Deploy Trial
See `sample_new_yaml.yml` for the yaml requirements for each trial. 

After setting up your .yaml file, run the following command to deploy a trial:

`python3 -m main filename.yml`

Notes: 
- If not `None`, ensure that your image_norm_csv has three values within two columns named `means` and  `stds` 
- Trials with the same ID, comb_config, and model will be overriden by the newest trial (by design)


### Directory Structure
After training at least one epoch, the following directory structure will be created to
be used for all future experiments. Note that all <CAPITALIZED> words are replaced with
their values from the configuration file. Briefly, all training sessions are considered
"trials" of an "experiment". Each experiment is named by its model type, the concatenation 
method utilized, and its train ID. You can also resume training a model by using its "args.yaml".

```
    Directory structure is
    <MAIN>/
    ├── datasets/
    │   └── <DATASET_NAME>/                        <-- Dataset metadata files (e.g. image statistics)
    └── trials/
        └── <MODEL_TYPE>_<COMB_METHOD>_<TRIAL_ID>/
            ├── checkpoints/
            │   └── best_model.pth                 <-- Best saved PyTorch model parameters
            ├── deployment_metadata/
            │   ├── dataset.csv                    <-- Dataset name, sampling indices, class names, size
            │   ├── image_statistics.csv           <-- Mean, std across all images from training set
            │   ├── image_metadata_statistics.csv  <-- Mean, std, max across all XMP metadata from training set
            │   └── training.csv                   <-- Early stopping count, epoch count
            ├── logs/
            │   ├── args.yaml                      <-- Saved config
            │   ├── output.txt                     <-- Printed output during trial
            │   ├── train_losses.txt               <-- Saved train losses per epoch 
            │   ├── valid_losses.txt               <-- Saved validation losses per epoch 
            │   ├── train_acc.txt                  <-- Saved train accuracies per epoch 
            │   ├── valid_acc.txt                  <-- Saved validation accuracies per epoch 
            │   └── loss_val_plot.png              <-- Plot of data from above .txt files 
            └── results/
                ├── results.png                    <-- Image for Sklearn summary chart, accuracies for all classes, balanced accuracy 
                └── confusion_matrix.png           <-- Confusion matrix image for test results 
```

## Results of Training

Models will stop training after the maximum number of epochs or once the`EarlyStopper` is activated. The `results/` directory will output the relevant .png files. 

## Resume Training 

Any `args.yml` file will automatically run resume-training mode. This mode will update results in the initial trial directory, but will dump two new files into `logs/`: resume_train_args.yml and resume_train_output.txt. This is to preserve the initial trial information and avoid overriding old output. Training will resume as usual. 


### Model / Checkpoint Save structure

Each saved model will include the following information: 
- number of classes
- p_dropout 
- comb_method
- comb_config
- neurons_reducer_block 
- mean values of the image transformations
- standard deviation values of the image transformations

if metadata is included: 
- standard deviation dictionary for the metadata training set 
- mean dictionary for the metadata training set 

### Meta Data CSV
**Note that file type at end assumed for all rows containing the image name (e.g. .jpg/.png)**

## Testing / Applying a trained model to unlabeled images
Setting `run_type` in your .yml file to `forward-inference` will sort images into folders given a pre-trained model. 
The path should be set to the file you want the images to be sorted into. This mode will create a folder for a class
if it is not existent in the path. 


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

## WSL Configuration 

Start here to activate WSL environment on your computer: https://learn.microsoft.com/en-us/windows/wsl/setup/environment
- Note: Information up to `Add additional distributions` is required, rest is helpful to know, but not necessary. 
- You can activate it anytime by opening your Command Prompt and simply typing `wsl`. 

Next, install MiniConda from the terminal: 

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
bash Miniconda3-latest-Linux-x86_64.sh
```

Now follow step 2 through step 4 here: https://wandb.ai/wandb/common-ml-errors/reports/How-To-Install-TensorFlow-With-GPU-Support-on-Windows--VmlldzozMDYxMDQ
- Note: Visual Studio needs to be installed on the Windows side (not in WSL) to install CUDA ToolKit. Here are the compatibilities listed on the NVIDIA site: https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html#cudnn-versions-windows. However, I had Visual Studio Community 2022 already installed. 
- Make sure CUDA ToolKit Version and cuDNN Package are compatible: https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html

### Torch Install 
Now for PyTorch, install on Linux through Conda in a new environment. Pick either CUDA 11.7 or 11.8 depending on what you installed earlier: https://pytorch.org/get-started/locally/


### Writing Code in WSL Environment 
If you want to run Jupyter though the WSL tunnel, use the following in the wsl terminal: `jupyter lab --no-browser` and copy the link into your browser. 

If you want to use VSCode, you need to tunnel into the WSL environment. This link has a more elegant method that I've never tried: https://code.visualstudio.com/docs/remote/wsl. However, I just went to the terminal and typed "WSL" and selected "WSL: Open folder in WSL". 