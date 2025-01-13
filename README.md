
# TAN-NILM
**GitHub Repository Address:** [https://github.com/zhaomaojiang/TAN-NILM](https://github.com/zhaomaojiang/TAN-NILM) If you have any questions, feel free to contact us at our email address: 1552751108@qq.com.

## File Introduction
-   `/check_appliance` stores the model parameters during the training process for each appliance.
-   `/data` contains data and data preprocessing-related code.
-   `/nnet` stores the network models.
-   `lossfunc.py` contains the loss function.
-   `train_appliance.py` is used to train the models for the corresponding appliances.
-   `test_appliance.py` is used to test the models for the corresponding appliances.

# Getting Started
Before running the code, you need to set up the Python environment dependencies and download the [REFIT dataset](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned), extracting it to the correct directory.

## Installing Dependencies
We use Python version 3.10.0 and CUDA version 12.1.  
Required dependencies:
-   `scipy==1.11.1`
-   `numpy==1.25.2`
-   `matplotlib==3.7.2`
-   `torch==2.1.0`
-   `torchaudio==2.1.0`
-   `torchvision==0.16.0`
-   `tqdm==4.66.1`
-   `pandas==2.0.3`

You can install the dependencies directly using pip:

`pip install -r environment.txt` 

Alternatively, you can create the environment precisely using Conda (a reliable network connection is required):

`conda env create -f environment.yml` 

## Downloading the Dataset
Before proceeding, download the `Processed_Data_CSV.7z` file from:  
[https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)

Extract the contents to the `./data/REFIT_dataset/REFIT/` directory.

## Running the Program

### Step 1: Process the Original Data
Run:

`cd ./data/REFIT_dataset/REFIT`
`python demo.py` 

This step converts the original CSV data format for each house into various files within their respective folders, including the `Channel.dat` file required for subsequent steps.

### Step 2: Generate Signatures
Run:

`cd .` to return to ./data/REFIT_dataset
Run:

`python gen_activations.py` 

This step generates the signature required for Step 3.

### Step 3: Generate Dataset
Run:

`python gen_dataset_by_activations.py` 

This step generates training and testing data, storing all data in a single pickle file.

### Step 4: Save Dataset in Batches

Run:

`python save_dataset_by_batch.py` 

To reduce memory usage, this step splits the dataset into batches and stores them.

### Step 5: Train or Test the Model

Run:

`cd ..` to return to the root directory.

then

`python ./train_appliance.py` 

or

`cd ..` to return to the root directory.

then

`python ./test_appliance.py` 

This step allows you to train or test the models for the corresponding appliances.

# Other Tips
There is a difference between linux and windws for data processing in 'DataSet'.  

DataLoader.remove_abnormal_data():

……

        mix_data = np.reshape(mix_data, (-1, 1))  # todo: for windows
        
        # mix_data = np.reshape(mix_data.to_numpy(), (-1, 1))  # for linux
        
        app_data = np.reshape(app_data, (-1, 1))  # todo: for windows
        
        # app_data = np.reshape(app_data.to_numpy(), (-1, 1))  # for linux
        
……
