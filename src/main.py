import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import yaml

from models.gnn import GNN
from training.train import load_data, train_model, validate_model, test_model
from utils.config import load_config

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    style_path = os.path.join(script_dir, 'styles/style.mplstyle')
    plt.style.use(style_path)
    config = load_config(os.path.join(script_dir, '..', 'config.yaml'))
    data_dir = config['data']['data_dir']
    results_dir = os.path.join(script_dir, '..', config['general']['save_dir'])
    plot = config['general']['plot']
    save = config['general']['save']
    mode = config['general']['mode']    
    #check for directory, if it doesn't exist, create it

    dataset = QM9(root = data_dir)
    train_loader, val_loader, test_loader, data_mean, data_std = load_data(data_dir = config['data']['data_dir'], 
                                                                data_size = config['data']['data_size'], 
                                                                train_fraction = config['data']['train_fraction'], 
                                                                val_fraction = config['data']['val_fraction'], 
                                                                test_fraction = config['data']['test_fraction'], 
                                                                batch_size = config['data']['batch_size'],
                                                                target_index = config['target']['index'])

    print(f"Train loader size: {len(train_loader.dataset)}")
    print(f"Validation loader size: {len(val_loader.dataset)}")
    print(f"Test loader size: {len(test_loader.dataset)}")

    model = GNN(input_dim = dataset.num_features, hidden_dim = config['model']['hidden_dim'], output_dim = config['model']['output_dim'])
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = config['training']['learning_rate'], weight_decay = 5e-4)

    if mode == "train":
        print("Training mode selected")

        if save:
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                print(f"Created directory: {results_dir}")
            else:
                print(f"Directory already exists: {results_dir}. Overwriting.")
                # exit()

        num_epochs = config['training']['num_epochs'] 

        train_loss = np.empty(num_epochs)
        val_loss = np.empty(num_epochs)

        train_target = np.empty(0)
        train_y_target = np.empty(0)

        best_loss = np.inf
        # patience_counter = 0
        # thresh = 0.005
        # patience = 3

        for epoch in range(num_epochs):
        #     #training
            epoch_loss, model = train_model(model, train_loader, criterion, optimizer)
            v_loss = validate_model(model, val_loader, criterion)

            if save and v_loss < best_loss:
                torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))

            train_loss[epoch] = epoch_loss.detach().numpy() 
            val_loss[epoch] = v_loss.detach().numpy()

            for d in train_loader:
                output = model(d)
                if epoch == num_epochs - 1:
                    #concatenates the predictions from the current batch into the train_target array 
                    train_target = np.concatenate((train_target, output.detach().numpy()[:,0])) 
                    #concatenates the true values from the current batch into the train_y_target array
                    train_y_target = np.concatenate((train_y_target, d.y.detach().numpy()))

            if epoch % 2 == 0:
                print(
                    "Epoch: "
                    + str(epoch)
                    + ", Train loss: "
                    + str(epoch_loss.item())
                    + ", Val loss: "
                    + str(v_loss.item())
                )

        #test model

        # test_loss, test_target, test_y_target = test_model(model, test_loader, criterion)
        

        if save:
            np.savetxt(os.path.join(results_dir, 'epoch_data.txt'), np.c_[train_loss, val_loss], header="train_loss, val_loss")
            np.savetxt(os.path.join(results_dir, 'target_data.txt'), np.c_[train_target, train_y_target], header="train_target, train_y_target")
            #save config file as text file
            with open(os.path.join(results_dir, 'config.txt'), 'w') as file:
                file.write(yaml.dump(config))

        if plot:
            fig, axs = plt.subplots(2, figsize=(5, 5))

            axs[0].plot(train_loss, label="train loss")
            axs[0].plot(val_loss, label="validation loss")
            axs[0].set_xlabel("Epoch") 
            axs[0].set_ylabel("Loss")
            axs[0].legend()

            axs[1].scatter(train_y_target, train_target, s = 1)
            axs[1].set_xlabel("True")
            axs[1].set_ylabel("Predicted")
            m = LinearRegression()
            m.fit(train_y_target.reshape(-1, 1), train_target)
            y_pred = m.predict(train_y_target.reshape(-1, 1))
            axs[1].plot(train_y_target, y_pred, color='red')
            axs[1].legend()
        
            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(results_dir, 'plots.png'))
            plt.show()

    elif mode == "test":
        print("Testing mode selected")
        #try to find the best model
        model_path = os.path.join(results_dir, 'best_model.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")

        model.load_state_dict(torch.load(model_path))

        test_loss, test_target, test_y_target = test_model(model, test_loader, criterion)

        print(f"Test loss: {test_loss.item()}")

        # denormalise the data 
        test_target = test_target * data_std + data_mean
        test_y_target = test_y_target * data_std + data_mean

        fig, ax = plt.subplots(figsize = (6,3))
        ax.scatter(test_y_target, test_target, s = 1)
        ax.set_xlabel("True / eV")
        ax.set_ylabel("Predicted / eV")
        m = LinearRegression()
        m.fit(test_y_target.reshape(-1, 1), test_target)
        y_pred = m.predict(test_y_target.reshape(-1, 1))
        r2 = r2_score(test_target, y_pred)
        ax.plot(test_y_target, y_pred, color='red', label = f"R2: {r2}")
        ax.legend()
        plt.tight_layout()
        plt.show()



