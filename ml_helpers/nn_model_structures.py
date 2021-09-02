import time

import numpy as np
import torch
import torch.nn as nn
import wandb


# Create a pytorch model

class RNN(nn.Module):
    '''
    Building the simple RNN model
    '''

    def __init__(self, RNN_nodes, linear_nodes, dropout_rate, input_size, n_output_vars):

        # Init the nn.module parent
        super(RNN, self).__init__()

        # Store input arguments
        self.RNN_nodes = RNN_nodes
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.linear_nodes = linear_nodes

        ### Now initialize the layers

        # LSTM 1
        # self.rnn = nn.LSTM(input_size=input_size, hidden_size=RNN_nodes, batch_first=True)
        # self.rnn = nn.GRU(input_size=input_size, hidden_size=RNN_nodes, batch_first=True)
        self.rnn = nn.RNN(input_size=input_size[0],
                          hidden_size=RNN_nodes,
                          batch_first=True,
                          nonlinearity='relu',
                          dropout=dropout_rate)
        # Batch_first gives input of shape (batch_size, timesteps, features)

        # Hidden 1
        self.hidden1 = nn.Linear(in_features=RNN_nodes + input_size[1], out_features=linear_nodes,
                                 bias=True)  # y = Ax + b

        # Hidden 2
        self.hidden2 = nn.Linear(in_features=linear_nodes, out_features=linear_nodes // 2, bias=True)

        # Output Layer
        self.out = nn.Linear(in_features=linear_nodes // 2, out_features=n_output_vars)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, X) -> torch.Tensor:
        ###     Run the network forward. This is evaluating the functions
        # X should be of shape (batch_size, timesteps, features)

        X_main = X[0]
        X_aux = X[1]

        # LSTM
        rnn_out_all_states, last_states = self.rnn(X_main)
        rnn_out = rnn_out_all_states[:, -1, :]  # State at t=-1

        # Concatenate Aux for t=0
        rnn_out_concat = torch.cat((rnn_out, X_aux), dim=1)

        # Hidden 1
        hidden_out = torch.relu(self.hidden1(rnn_out_concat))

        # Hidden 1
        hidden_out = torch.relu(self.hidden2(hidden_out))

        # Output Layer
        output = self.out(hidden_out)

        return output

    def train_model(self, train_features_norm, train_labels_norm, val_features_norm, val_labels_norm,
                    batch_size, train_batch_indexes):

        # optimizer = torch.optim.AdamW(params=self.parameters(), lr=1e-2, weight_decay=1e-1)
        optimizer = torch.optim.SGD(params=self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-4, 1e-2, mode='exp_range')
        loss_funct = torch.nn.MSELoss()

        # Convert data to tensors
        train_features_norm = [torch.from_numpy(train_features_norm[i].astype(np.float32)).cuda() for i in
                               range(len(train_features_norm))]
        val_features_norm = [torch.from_numpy(val_features_norm[i].astype(np.float32)).cuda() for i in
                             range(len(train_features_norm))]
        train_labels_norm = torch.from_numpy(train_labels_norm.astype(np.float32)).cuda()
        val_labels_norm = torch.from_numpy(val_labels_norm.astype(np.float32)).cuda()

        # Send model to device
        dev_name = 'cuda:0'  # Using GPU
        device = torch.device(dev_name)
        model = self.to(device)

        train_iter = cum_loss = report_loss = 0
        cum_examples = report_batches = 0
        begin_time = time.time()

        best_val_loss = 100
        best_counter = 0
        NUM_EPOCHS = 500
        for epoch in range(NUM_EPOCHS):
            model.train()
            for batch_index in train_batch_indexes[:-1]:
                train_iter += 1
                # zero the parameter gradients
                optimizer.zero_grad()

                # Get data for batch
                batch_features = [train_features_norm[i][batch_index] for i in range(len(train_features_norm))]
                batch_labels = train_labels_norm[batch_index]

                y = model(batch_features)  # forward
                batch_loss = loss_funct(y, batch_labels)  # forward

                batch_loss.backward()  # compute gradients

                optimizer.step()  # apply gradients
                scheduler.step()  # Step cycling learning rate
                batch_losses_val = batch_loss.item()
                report_loss += batch_losses_val
                cum_loss += batch_losses_val
                report_batches += 1
                cum_examples += batch_size

            # Validate and Output
            model.eval()
            val_loss = loss_funct(model(val_features_norm), val_labels_norm)
            print('epoch %d, avg. loss %.6f, val. loss %.6f ' \
                  'time elapsed %.2f sec' % (epoch + 1,
                                             report_loss / report_batches, val_loss,
                                             time.time() - begin_time), end=", ")
            wandb.log({'mse_train': report_loss / report_batches,
                       'mse_val': val_loss,
                       'epoch': epoch})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_counter = 0
            else:
                best_counter += 1

                if best_counter == 20:
                    break
            print('Counter: ', best_counter)
            train_time = time.time()
            report_loss = report_batches = 0.

        return model


class MLP(nn.Module):
    '''
    Building the simple RNN model
    '''

    def __init__(self, linear_nodes=1, dropout_rate=1.,
                 input_size=1, n_output_vars=1):
        # Init the nn.module parent
        super(MLP, self).__init__()

        # Store input arguments
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.linear_nodes = linear_nodes

        ### Now initialize the layers

        self.flatten = nn.Flatten()

        # Hidden 1
        self.hidden1 = nn.Linear(in_features=input_size, out_features=linear_nodes, bias=True)  # y = Ax + b
        # Hidden 2
        self.hidden2 = nn.Linear(in_features=linear_nodes, out_features=linear_nodes, bias=True)

        # Output Layer
        self.out = nn.Linear(in_features=linear_nodes, out_features=n_output_vars)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, X) -> torch.Tensor:
        ###     Run the network forward. This is evaluating the functions
        # X should be of shape (batch_size, timesteps, features)

        X = self.flatten(X)

        # Hidden 1
        hidden_out = torch.relu(self.hidden1(X))

        # Hidden 1
        hidden_out = torch.relu(self.hidden2(hidden_out))

        # Output Layer
        output = self.out(hidden_out)

        return output


def train(model, train_features_norm, train_labels_norm, val_features_norm, val_labels_norm, batch_size,
          train_batch_indexes):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=.01)
    loss_funct = torch.nn.MSELoss()

    # Convert data to tensors
    train_features_norm = torch.from_numpy(train_features_norm.astype(np.float32)).cuda()
    val_features_norm = torch.from_numpy(val_features_norm.astype(np.float32)).cuda()
    train_labels_norm = torch.from_numpy(train_labels_norm.astype(np.float32)).cuda()
    val_labels_norm = torch.from_numpy(val_labels_norm.astype(np.float32)).cuda()

    # Send model to device
    dev_name = 'cuda:0'  # Using GPU
    device = torch.device(dev_name)
    model = model.to(device)

    train_iter = cum_loss = report_loss = 0
    cum_examples = report_batches = 0
    begin_time = time.time()
    best_val_loss = 100
    best_counter = 0
    NUM_EPOCHS = 500

    for epoch in range(NUM_EPOCHS):

        for batch_index in train_batch_indexes[:-1]:
            train_iter += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # Get data for batch
            batch_features, batch_labels = train_features_norm[batch_index], train_labels_norm[batch_index]

            y = model(batch_features)  # forward
            batch_loss = loss_funct(y, batch_labels)  # forward

            batch_loss.backward()  # compute gradients

            optimizer.step()  # apply gradients

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
            report_batches += 1
            cum_examples += batch_size

        # Validate and Output
        val_loss = loss_funct(model(val_features_norm), val_labels_norm)

        print('epoch %d, avg. loss %.6f, val. loss %.6f ' \
              'time elapsed %.2f sec' % (epoch + 1,
                                         report_loss / report_batches, val_loss,
                                         time.time() - begin_time), end=" ")
        wandb.log({'mse_train': report_loss / report_batches,
                   'mse_val': val_loss,
                   'epoch': epoch})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_counter = 0
        else:
            best_counter += 1

            if best_counter == 20:
                break
        print('Counter: ', best_counter)

        train_time = time.time()
        report_loss = report_batches = 0.

    return model
