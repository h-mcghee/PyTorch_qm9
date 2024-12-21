# GNN QM9 Project

Building and training a Graph Neural Network (GNN) using the QM9 dataset for molecular property prediction. Easy to configure code for training, testing and visualising model performance.

## Usage

Run the `main.py` script to train or test the GNN model

```
python main.py
```

### Configuration

All setting and hyperparameters are toggled in the config.yaml file.

To train the model, set the mode to "train"

This will train the model according to the selected hyperparameters. 

The best model, epoch training / validation losses, and true vs predicted training values (for the final epoch) are saved into a directory specified by "save_dir". If directory already exists the code will overwrite.

Note, this will only save if "save" is set to true

Testing:

Set the mode to "test"

This will load the best saved model, evaluate the model on the test dataset, and display the results with R-2 plots.
