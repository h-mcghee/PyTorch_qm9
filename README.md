# GNN QM9 Project

This repository contains a Graph Neural Network (GNN) implementation for predicting molecular properties from the QM9 dataset using PyTorch Geometric. The code supports both training and testing modes, with configurable parameters defined in a YAML configuration file.

## Features

* Implements a Graph Neural Network (GNN) with three graph convolutional layers (GCNConv).
* Uses the QM9 dataset to predict molecular properties
* Provides visualisation of training and validation loss, as well as true vs predicted values
* Configurable through a config.yaml fild
* Saves the best model during training.

## Installation

1. Clone the repository:

```
git clone https://github.com/h-mcghee/PyTorch_qm9
```

2. Navigate to project directory

```
cd <your_directory> 
```

3. Create and activate a virtual environment:

```
python -m venv venv 
source venv/bin/activate
```

4. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

**Note:** The QM9 dataset is required for this project but is ignored in `.gitignore`. The dataset will automatically download when you run the `main.py` script for the first time. It will write to the `data_dir` path specified in `config.yaml`.

### Training

1. Ensure `mode` is set to `train` in `config.yaml`
2. Specify the `save_dir` where results are saved (if it already exists it will overwrite).
3. Modify hyperparameters and settings in `config.yaml`
4. Run the script:
   ```
   python src/main.py
   ```
5. The training process will:
   * Print epoch training and validation losses.
   * Save the best model to the directory specified in `save_dir`
   * Generate a plot at the end of training (epoch vs loss, and true vs predicted training set values for the final epoch)

### Testing

1. Ensure `mode` is set to `test` in `config.yaml`
2. Run the script:

   ```
   python src/main.py
   ```
3. The testing process will:

   * Load the best model from `save_dir`
   * Evaluate the model on the test dataset
   * Generate true vs predicted scatter plots and display the R^2 score
