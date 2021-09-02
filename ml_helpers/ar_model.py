import numpy as np
import pickle
import sklearn.preprocessing
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.multioutput
import torch
import time
import wandb
import xgboost as xgb

import MLPrograms
import nn_model_structures


class ARModel:

    def __init__(self, input_df, label_cols=None, drop_cols=None, n_out=2, n_in = 16, nonsequential_features=3):
        '''
        Class
        :param input_df: Whole Dataframe to train
        :param n_out: Number of forecast steps (including 0)
        :param n_in:  Number of lagging steps
        :param nonsequential_features: Number of nonsequential features in input_df that should not be in the RNN layer
        '''
        self.n_out = n_out  # Forecasting steps
        self.n_in = n_in  # Past Steps
        self.df = input_df
        self.nonsequential_features = nonsequential_features
        self.label_cols = label_cols
        self.drop_cols = drop_cols


    def get_training_data(self, df=None, label_cols=None, drop_cols=None, columns_to_keep=None, n_in=None, n_out=None, return_df=False, shuffle=True):
        '''
        Returns features, labels, n_vars for the input df. Must specify label_cols and drop_cols to separate out features from labels.
        :param df: Whether to specifiy an input df
        :param label_cols: Column names that act as labels. Please append f"(t+{i})" for each timestep required to predict.
        :param drop_cols: Column names to drop.  Please append f"(t+{i})" for each timestep required to drop.
        :param columns_to_keep: Optional. Whether to use only a subset of the columns in the original df
        :param n_in: Number of lagging steps (Default: self.n_in)
        :param n_out: Number of forecasting steps (Default: self.n_out)
        :param return_df: Whether to return a dataframe instead of numpy array. Useful for selecting features and label cols. (Default: False)
        :param shuffle: Whether to shuffle the timeseries data in batches.
        :return:
        '''
        if n_in is None:
            n_in = self.n_in
        if n_out is None:
            n_out = self.n_out

        if df is None:
            df = self.df

        if label_cols is None:
            label_cols = self.label_cols
        if drop_cols is None:
            drop_cols = self.drop_cols

        if label_cols is None:
            print('Warning: No label cols specified, will return an empty dataframe')
        if drop_cols is None:
            print('Warning: No drop cols specified, will treat all n_out future data as known and included in the features. May give unrealistic results.')

        if columns_to_keep is not None:
            df = df[columns_to_keep]

        # Reshapes data to a dataframe containing lagging and future observations in the same timestep
        data_sup, n_vars = MLPrograms.series_to_supervised(df, n_out=n_out,
                                                           n_in=n_in)  # Gets previous timesteps as inputs

        features_df = data_sup.drop(columns=drop_cols + label_cols).dropna()
        labels_df = data_sup[label_cols]
        if return_df:
            return features_df, labels_df

        labels = labels_df.values
        features = features_df.values

        # Shuffle the datasets in batches (important to use batches to not bias timeseries prediction)
        if shuffle:
            features, labels = MLPrograms.batch_shuffle((features, labels), shuffle_size=512)

        return features, labels, n_vars

    def encode(self, features, labels, model_type=None, n_vars=None, n_in=None, rescale=False, savescale=False):
        '''
        Scales the input data and optionally reshapes the data into an RNN
        :param features: np array of features
        :param labels: np array of labels
        :param model_type: which ml model type to use
        :param n_vars: number of features
        :param n_in: number of forecasting steps
        :param rescale: Whether to rescale the data. One encoding operation must be ran to initialize the scalers.
        :param savescale: Optional whether to save the scalers. Not implemented yet.
        :return:
        '''
        if model_type is None: model_type = self.model_type
        if n_vars is None: n_vars = self.n_vars
        if n_in is None: n_in = self.n_in

        # Normalize Data
        if rescale:
            self.f_scaler = sklearn.preprocessing.StandardScaler()
            features_norm = self.f_scaler.fit_transform(features)

            self.l_scaler = sklearn.preprocessing.StandardScaler()
            labels_norm = self.l_scaler.fit_transform(labels)
        else:
            try:
                features_norm = self.f_scaler.transform(features)
                labels_norm = self.l_scaler.transform(labels)
            except Exception as e:
                print(e)
                ValueError("Scaler is likely not defined, call encode with rescale=True.")

        if model_type == 'rnn':
            if self.nonsequential_features > 0:
                # Reshape into (batch, timesteps, features) and get aux features
                features_norm_main, features_norm_aux = features_norm[:, :-n_vars + self.nonsequential_features], features_norm[:, -n_vars + self.nonsequential_features:]

                features_norm_main = features_norm_main.reshape((features_norm_main.shape[0], n_in, n_vars))

                features_norm = (features_norm_main, features_norm_aux)
            else:
                features_norm = features_norm.reshape((features_norm.shape[0], n_in, n_vars))

        return features_norm, labels_norm

    def train_models(self, model_type, train_size=.75, wandb_project_name=''):
        '''
        Trains the model using the input df and other information specified during instantiation.
        :param model_type: Type of model. ['rnn', 'mlp', 'xgb', 'linear']]
        :param train_size: Size of the training dataset for the train/val split
        :param wandb_project_name: If specified, will log the results to wandb
        :return:
        '''
        np.random.seed(0)
        # Params
        self.model_type = model_type
        self.batch_size = 128

        features, labels, n_vars = self.get_training_data()
        self.n_vars = n_vars

        train_features, val_features = features[:int(train_size * features.shape[0]), :], features[int(train_size
                                                                                                       * features.shape[0]):, :]
        train_labels, val_labels = labels[:int(train_size * labels.shape[0]), :], labels[int(train_size
                                                                                             * labels.shape[0]):, :]

        train_features_norm, train_labels_norm = self.encode(train_features, train_labels, rescale=True, savescale=True)
        val_features_norm, val_labels_norm = self.encode(val_features, val_labels, rescale=True, savescale=True)

        # Generate indexes to batch data and shuffle
        train_batch_indexes = [range(i, i + self.batch_size) for i in
                               range(0, train_features_norm.shape[0], self.batch_size)]
        train_batch_indexes = [train_batch_indexes[i] for i in
                               np.random.choice(range(len(train_batch_indexes) - 1),
                                                size=len(train_batch_indexes) - 1,
                                                replace=False)]

        ##############################  Build Model  ##############################

        if wandb_project_name != '':
            wandb.init(project=wandb_project_name, reinit=True)

        start_time = time.time()
        if model_type == 'rnn':
            model, train_preds_norm, val_preds_norm = self._train_rnn(train_features_norm, train_labels_norm,
                                                                      val_features_norm, val_labels_norm,
                                                                      train_batch_indexes
                                                                      )
        elif model_type == 'mlp':
            model, train_preds_norm, val_preds_norm = self._train_mlp(train_features_norm, train_labels_norm,
                                                                      val_features_norm, val_labels_norm,
                                                                      train_batch_indexes
                                                                      )
        elif model_type == 'linear':
            model, train_preds_norm, val_preds_norm = self._train_linear(train_features_norm, train_labels_norm,
                                                                         val_features_norm, val_labels_norm,
                                                                         train_batch_indexes
                                                                         )

        elif model_type == 'xgb':
            model, train_preds_norm, val_preds_norm = self._train_xgb(train_features_norm, train_labels_norm,
                                                                      val_features_norm, val_labels_norm,
                                                                      train_batch_indexes
                                                                      )
        else:
            raise(ValueError(f"'{model_type}' not a valid model."))

        train_error = np.mean((train_preds_norm - train_labels_norm) ** 2, axis=0)
        val_error = np.mean((val_preds_norm - val_labels_norm) ** 2, axis=0)
        print('Train Error ', train_error, np.mean(train_error))
        print('Val Error ', val_error, np.mean(val_error))
        wandb.log({'Tot Train Error': np.mean(train_error),
                   'Var Train Error': train_error,
                   'Tot Val Error': np.mean(val_error),
                   'Var Val Error': val_error,
                   'Model Type': model_type,
                   'Train Time': time.time() - start_time
                   })

    def _train_rnn(self, train_features_norm, train_labels_norm, val_features_norm, val_labels_norm, train_batch_indexes):
        '''
        Must build a rnn in the nn_model_structures module.
        :param train_features_norm:
        :param train_labels_norm:
        :param val_features_norm:
        :param val_labels_norm:
        :param train_batch_indexes:
        :return:
        '''
        model = nn_model_structures.RNN(RNN_nodes=64,
                                        linear_nodes=32,
                                        dropout_rate=.2,
                                        input_size=(train_features_norm[0].shape[2], train_features_norm[1].shape[1]),
                                        n_output_vars=train_labels_norm.shape[1])
        model = model.train_model(train_features_norm, train_labels_norm,
                                  val_features_norm, val_labels_norm,
                                  self.batch_size, train_batch_indexes)
        model.n_in = self.n_in
        model.n_out = self.n_out
        model.eval()
        val_preds_norm = model((
            torch.from_numpy(val_features_norm[0].astype(np.float32)).cuda(),
            torch.from_numpy(val_features_norm[1].astype(np.float32)).cuda())).cpu().detach().numpy()
        train_preds_norm = model((
            torch.from_numpy(train_features_norm[0].astype(np.float32)).cuda(),
            torch.from_numpy(train_features_norm[1].astype(np.float32)).cuda())).cpu().detach().numpy()

        # torch.save(model, f'{self.model_dir}/rnn/model_{self.model_num}.torch')

        return model, train_preds_norm, val_preds_norm

    def _train_mlp(self, train_features_norm, train_labels_norm, val_features_norm, val_labels_norm, train_batch_indexes):
        model = nn_model_structures.MLP(linear_nodes=32,
                                        dropout_rate=.2,
                                        input_size=train_features_norm.shape[1],
                                        n_output_vars=val_labels_norm.shape[1])
        model = nn_model_structures.train(model, train_labels_norm, val_labels_norm,
                                          train_features_norm, val_features_norm,
                                          self.batch_size, train_batch_indexes)
        model.eval()
        val_preds_norm = model(
            torch.from_numpy(val_features_norm.astype(np.float32)).cuda()).cpu().detach().numpy()
        train_preds_norm = model(
            torch.from_numpy(train_features_norm.astype(np.float32)).cuda()).cpu().detach().numpy()
        # torch.save(model, f'{self.model_dir}/mlp/model_{self.model_num}.torch')

        return model, train_preds_norm, val_preds_norm

    def _train_linear(self, train_features_norm, train_labels_norm, val_features_norm, val_labels_norm, train_batch_indexes):
        X = train_features_norm.reshape(train_features_norm.shape[0], -1)
        y = train_labels_norm

        model = sklearn.linear_model.LinearRegression().fit(X, y)

        val_preds_norm = model.predict(val_features_norm)
        train_preds_norm = model.predict(train_features_norm)
        # pickle.dump(model, open(f'{self.model_dir}/linear/model_{self.model_num}.mod', 'wb'))

        return model, train_preds_norm, val_preds_norm

    def _train_xgb(self, train_features_norm, train_labels_norm, val_features_norm, val_labels_norm, train_batch_indexes):
        xgb_params = {
            'tree_method': 'gpu_hist',
            'max_depth': 3,
            'eta': .1,
            'objective': 'reg:squarederror',
            'min_child_weight': 6,
            'subsample': .9863,
            # 'colsample_bytree': config.colsample_bytree,
            'eval_metric': 'rmse',
        }
        X = train_features_norm
        y = train_labels_norm

        model = sklearn.multioutput.MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror',
                                                                          tree_method='gpu_hist')).fit(X, y)
        print('Train Error ', np.mean((model.predict(X) - y) ** 2))
        print('Val Error ', np.mean((model.predict(val_features_norm) - val_labels_norm) ** 2))

        val_preds_norm = model.predict(val_features_norm)
        train_preds_norm = model.predict(train_features_norm)
        # pickle.dump(model, open(f'{self.model_dir}/xgb/model_{self.model_num}.mod', 'wb'))

        return model, train_preds_norm, val_preds_norm
