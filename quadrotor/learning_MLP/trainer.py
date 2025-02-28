import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
from scipy.io import loadmat

from learning_MLP.core import Core


class Regression_Trainer(Core):
    def __init__(self, model_res, param_reg):
        super(Regression_Trainer, self).__init__()
        self.load_labelfeature()
        self.device = torch.device("cuda")
        self.model_res = model_res
        self.param_reg = param_reg

    def load_labelfeature(self):
        # dir_ = self.frame_dir + "/data/"
        dir_ = "data"
        s1fname_list = [file for file in os.listdir(dir_) if file.endswith(".mat")]
        X_dataset = []
        Y_dataset = []

        for s1fname in s1fname_list:
            comp_data = loadmat(dir_ + "/" + s1fname)
            xk_real_seq = comp_data["xk_real_seq"]
            drag_seq = comp_data["drag_seq"]
            # Pick only [vel, eul] as state input
            X_dataset += [xk_real_seq[3:9, :]]
            Y_dataset += [drag_seq]
            # X: 3+3 x N
            # Y: 3 x N

        self.X_dataset = np.hstack(X_dataset).T  # Transpose for torch training
        self.Y_dataset = np.hstack(Y_dataset).T  # Transpose for torch training
        # print(self.X_dataset.shape, self.Y_dataset.shape)

    def split_dataset(self, batch_size, ratio=[8, 1, 1]):
        X_tensor = torch.tensor(self.X_dataset, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(self.Y_dataset, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, Y_tensor)
        dataset_len = len(self.X_dataset)
        train_size = int(dataset_len * ratio[0] / 10)
        val_size = int(dataset_len * ratio[1] / 10)
        test_size = dataset_len - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # val_loader = []
        # test_loader = []
        
        return train_loader, val_loader, test_loader
    
    def loss_regression(self, model, inputs, labels):
        # Get outputs to device
        outputs = model(inputs).to(self.device)
        # Define Loss
        loss_obj = torch.nn.MSELoss()
        loss_mse = loss_obj(outputs, labels).to(self.device)
        # Define L1 loss
        l1_reg = torch.tensor(0.0, requires_grad=True).to(self.device)
        for param in model.parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        loss = loss_mse + self.param_reg * l1_reg
        return loss, loss_mse

    def optimize_minibatch(
        self,
        learning_rate,
        batch_size,
        epochs,
    ):
        # Setup
        model = self.model_res()
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_seq = []

        # Get data
        train_loader, val_loader, test_loader = self.split_dataset(batch_size)

        # Record Run
        self.record_run("minibatch", "torchADAM")

        for ii in range(epochs):
            model.train()
            train_loss = 0.0
            btch_id = 1

            # Go throught the train batches
            for batch_inputs, batch_labels in train_loader:
                # print(batch_inputs.shape, batch_labels.shape)
                batch_labels.to(self.device)
                batch_labels.to(self.device)
                # get loss and compute grad
                optimizer.zero_grad()
                loss_, loss_mse = self.loss_regression(
                    model, batch_inputs, batch_labels
                )
                loss_.backward()
                # Iterate
                optimizer.step()
                # Add loss to list
                train_loss += loss_mse.item()
                # Display Minibatch Progress
                print(
                    f"\rEpoch [{ii+1}/{epochs}] ",
                    "BatchProgress: {0}{1}% ".format(
                        "▉" * int(btch_id / len(train_loader) * 100 / 10),
                        int(btch_id / len(train_loader) * 100),
                    ),
                    end="",
                )
                btch_id += 1

            # Go through the val and test batches
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_inputs, batch_labels in val_loader:
                    loss_, loss_mse = self.loss_regression(
                        model, batch_inputs, batch_labels
                    )
                    val_loss += loss_mse.item()

            test_loss = 0.0
            with torch.no_grad():
                for batch_inputs, batch_labels in test_loader:
                    loss_, loss_mse = self.loss_regression(
                        model, batch_inputs, batch_labels
                    )
                    test_loss += loss_mse.item()

            # Reform loss
            if train_loader: train_loss /= len(train_loader)
            if val_loader: val_loss /= len(val_loader)
            if test_loader: test_loss /= len(test_loader)
            loss_seq += [np.array([train_loss, val_loss, test_loss])]
            # Display Epoch Progress
            print(
                f"Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, Test Loss: {test_loss:.4e}"
            )

            # Save Epoch and Model
            self.record_loss(np.vstack(loss_seq))
            self.save_model(model, epoch=ii + 1)

        # Output model params as vector for stage 2 usage
        params = self.get_modelparam_vec(model_param_gen=model.parameters())

        return params
