#----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.optim import Adamax as optimm

import numpy as np
import time
import itertools
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics.pairwise import euclidean_distances
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import collections

def _np_to_tensor(arr):
    arr = np.array(arr, dtype='f4')
    arr = torch.from_numpy(arr)
    return arr

class NLS(BaseEstimator):
    """
    Estimate a neuro local smoother (NLS).
    Parameters
    ----------
    ncomponents : integer
        Maximum number of components of the Fourier series
        expansion.
    nn_weight_decay : object
        Mulplier for penalizaing the size of neural network weights. This penalization occurs for training only (does not affect score estimator nor validation of early stopping).
    num_layers : integer
        Number of hidden layers for the neural network. If set to 0, then it degenerates to linear regression.
    hidden_size : integer
        Multiplier for the size of the hidden layers of the neural network. If set to 1, then each of them will have ncomponents components. If set to 2, then 2 * ncomponents components, and so on.
    es : bool
        If true, then will split the training set into training and validation and calculate the validation internally on each epoch and check if the validation loss increases or not.
    es_validation_set_size : float, int
        Size of the validation set if es == True, given as proportion of train set or as absolute number. If None, then `round(min(x_train.shape[0] * 0.10, 5000))` will be used.
n_train = x_train.shape[0] - n_test
    es_give_up_after_nepochs : float
        Amount of epochs to try to decrease the validation loss before giving up and stoping training.
    es_splitter_random_state : float
        Random state to split the dataset into training and validation.
    nepoch : integer
        Number of epochs to run. Ignored if es == True.
    batch_initial : integer
        Initial batch size.
    batch_step_multiplier : float
        See batch_inital.
    batch_step_epoch_expon : float
        See batch_inital.
    batch_max_size : float
        See batch_inital.
    batch_test_size : integer
        Size of the batch for validation and score estimators.
        Does not affect training efficiency, usefull when there's
        little GPU memory.
    gpu : bool
        If true, will use gpu for computation, if available.
    verbose : integer
        Level verbosity. Set to 0 for silent mode.
    """
    def __init__(self,
                 nn_weight_decay=0,
                 num_layers=3,
                 hidden_size=100,

                 es = True,
                 es_validation_set_size = None,
                 es_give_up_after_nepochs = 30,
                 es_splitter_random_state = 0,

                 nepoch=200,

                 batch_initial=300,
                 batch_step_multiplier=1.4,
                 batch_step_epoch_expon=2.0,
                 batch_max_size=1000,

                 optim_lr=1e-3,

                 dataloader_workers=1,
                 batch_test_size=2000,
                 gpu=True,
                 verbose=1,

                 scale_data=True,
                 varying_theta0=True,
                 fixed_theta0=True,

                 penalization_thetas=0,
                 penalization_variable_theta0=0,

                 n_classification_labels=0,
                 ):

        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def fit(self, x_train, y_train):
        if self.scale_data:
            self.scaler = StandardScaler()
            self.scaler.fit(x_train)
        self.gpu = self.gpu and torch.cuda.is_available()

        self.x_dim = x_train.shape[1]
        if len(y_train.shape) == 1:
            y_train = y_train[:, None]
        self.y_dim = y_train.shape[1]

        self._construct_neural_net()
        self.epoch_count = 0

        if self.gpu:
            self.move_to_gpu()

        return self.improve_fit(x_train, y_train, self.nepoch)

    def move_to_gpu(self):
        self.neural_net.cuda()
        self.gpu = True

        return self

    def move_to_cpu(self):
        self.neural_net.cpu()
        self.gpu = False

        return self

    def improve_fit(self, x_train, y_train, nepoch=1):
        if len(y_train.shape) == 1:
            y_train = y_train[:, None]

        if self.scale_data:
            x_train = self.scaler.transform(x_train)

        assert(self.batch_initial >= 1)
        assert(self.batch_step_multiplier > 0)
        assert(self.batch_step_epoch_expon > 0)
        assert(self.batch_max_size >= 1)
        assert(self.batch_test_size >= 1)

        assert(self.num_layers >= 0)
        assert(self.hidden_size > 0)

        if self.n_classification_labels:
            y_dtype = np.int64
            criterion = nn.CrossEntropyLoss()
        else:
            y_dtype = "f4"
            criterion = nn.MSELoss()

        inputv_train = np.array(x_train, dtype='f4')
        target_train = np.array(y_train, dtype=y_dtype)

        range_epoch = range(nepoch)
        if self.es:
            es_validation_set_size = self.es_validation_set_size
            if es_validation_set_size is None:
                es_validation_set_size = round(
                    min(x_train.shape[0] * 0.10, 5000))
            splitter = ShuffleSplit(n_splits=1,
                test_size=es_validation_set_size,
                random_state=self.es_splitter_random_state)
            index_train, index_val = next(iter(splitter.split(x_train,
                y_train)))
            self.index_train = index_train
            self.index_val = index_val

            inputv_val = inputv_train[index_val]
            target_val = target_train[index_val]
            inputv_val = np.ascontiguousarray(inputv_val)
            target_val = np.ascontiguousarray(target_val)

            inputv_train = inputv_train[index_train]
            target_train = target_train[index_train]
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            self.best_loss_val = np.infty
            es_tries = 0
            range_epoch = itertools.count() # infty iterator

            batch_test_size = min(self.batch_test_size,
                                  inputv_val.shape[0])
            self.loss_history_validation = []

        batch_max_size = min(self.batch_max_size, inputv_train.shape[0])
        self.loss_history_train = []

        start_time = time.time()

        self.actual_optim_lr = self.optim_lr
        optimizer = optimm(
            self.neural_net.parameters(),
            lr=self.actual_optim_lr,
            weight_decay=self.nn_weight_decay
        )
        err_count = 0
        es_penal_tries = 0
        for _ in range_epoch:
            self.cur_batch_size = int(min(batch_max_size,
                self.batch_initial +
                self.batch_step_multiplier *
                self.epoch_count ** self.batch_step_epoch_expon))

            permutation = np.random.permutation(target_train.shape[0])
            inputv_train = torch.from_numpy(inputv_train[permutation])
            target_train = torch.from_numpy(target_train[permutation])
            inputv_train = np.ascontiguousarray(inputv_train)
            target_train = np.ascontiguousarray(target_train)

            try:
                self.neural_net.train()
                self._one_epoch(True, self.cur_batch_size, inputv_train,
                                target_train, optimizer, criterion)

                if self.es:
                    self.neural_net.eval()
                    avloss = self._one_epoch(False, batch_test_size,
                        inputv_val, target_val, optimizer,
                        criterion)
                    self.loss_history_validation.append(avloss)
                    if avloss <= self.best_loss_val:
                        self.best_loss_val = avloss
                        best_state_dict = self.neural_net.state_dict()
                        best_state_dict = deepcopy(best_state_dict)
                        es_tries = 0
                        if self.verbose >= 2:
                            print("This is the lowest validation loss",
                                  "so far.")
                        self.best_loss_history_validation = avloss
                    else:
                        es_tries += 1

                    if (es_tries == self.es_give_up_after_nepochs
                        // 3 or
                        es_tries == self.es_give_up_after_nepochs
                        // 3 * 2):
                        if self.verbose >= 2:
                            print("No improvement for", es_tries,
                             "tries")
                            print("Decreasing learning rate by half")
                            print("Restarting from best route.")
                        optimizer.param_groups[0]['lr'] *= 0.5
                        self.neural_net.load_state_dict(
                            best_state_dict)
                    elif es_tries >= self.es_give_up_after_nepochs:
                        self.neural_net.load_state_dict(
                            best_state_dict)
                        if self.verbose >= 1:
                            print(
                                "Validation loss did not improve after",
                                self.es_give_up_after_nepochs,
                                "tries. Stopping"
                            )
                        break

                self.epoch_count += 1
            except RuntimeError as err:
                #if self.epoch_count == 0:
                #    raise err
                if self.verbose >= 2:
                    print("Runtime error problem probably due to",
                           "high learning rate.")
                    print("Decreasing learning rate by half.")

                self._construct_neural_net()
                if self.gpu:
                    self.move_to_gpu()
                self.actual_optim_lr /= 2
                optimizer = optimm(
                    self.neural_net.parameters(),
                    lr=self.actual_optim_lr,
                    weight_decay=self.nn_weight_decay
                )
                self.epoch_count = 0

                continue
            except KeyboardInterrupt:
                if self.epoch_count > 0 and self.es:
                    print("Keyboard interrupt detected.",
                          "Switching weights to lowest validation loss",
                          "and exiting")
                    self.neural_net.load_state_dict(best_state_dict)
                break

        self.elapsed_time = time.time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", self.elapsed_time, flush=True)

        return self

    def _one_epoch(self, is_train, batch_size, inputv, target,
        optimizer, criterion):
        #with torch.set_grad_enabled(is_train):
            inputv = torch.from_numpy(inputv)
            target = torch.from_numpy(target)

            loss_vals = []
            batch_sizes = []

            tdataset = data.TensorDataset(inputv, target)
            data_loader = data.DataLoader(tdataset,
                batch_size=batch_size, shuffle=True, drop_last=is_train,
                pin_memory=self.gpu,
                num_workers=self.dataloader_workers,)

            for inputv_this, target_this in data_loader:
                if self.gpu:
                    inputv_this = inputv_this.cuda(non_blocking=True)
                    target_this = target_this.cuda(non_blocking=True)

                inputv_this.requires_grad_(True)

                batch_actual_size = inputv_this.shape[0]

                optimizer.zero_grad()
                theta0v, theta0f, thetas = self.neural_net(inputv_this)

                inputs_ext = inputv_this
                if self.n_classification_labels:
                    inputs_ext = inputv_this[..., None]

                output = (thetas * inputs_ext).sum(1, True)

                if theta0v is not None:
                    output = output + theta0v
                if theta0f is not None:
                    output = output + theta0f

                if self.n_classification_labels:
                    output = output.transpose(1,2)

                loss = criterion(output, target_this)

                # Derivative penalization start
                if not self.n_classification_labels:
                    thetas = thetas[..., None]
                    if theta0v is not None:
                        theta0v = theta0v[..., None]

                if self.penalization_thetas:
                    for i in range(thetas.shape[1]):
                        for j in range(thetas.shape[2]):
                            grads_this, = torch.autograd.grad(
                                thetas[:, i, j].sum(), inputv_this,
                                create_graph=True,
                                )
                            loss_this = (grads_this**2).mean(0).sum()
                            if i or j:
                                loss2 = loss2 + loss_this
                            else:
                                loss2 = loss_this

                    loss2 = loss2 * self.penalization_thetas
                    loss = loss + loss2

                if (self.penalization_variable_theta0
                    and theta0v is not None):
                    for i in range(theta0v.shape[1]):
                        grads_this, = torch.autograd.grad(
                                theta0v[i].sum(), inputv_this,
                                create_graph=True,
                                )

                        loss_this = (grads_this ** 2).mean(0).sum()
                        loss_this = loss_this * (
                            self.penalization_variable_theta0)
                        if i:
                                loss3 = loss3 + loss_this
                        else:
                                loss3 = loss_this
                    loss = loss + loss3
                # Derivative penalization end

                np_loss = loss.data.item()
                if np.isnan(np_loss):
                    raise RuntimeError("Loss is NaN")

                loss_vals.append(np_loss)
                batch_sizes.append(batch_actual_size)

                if is_train:
                    loss.backward()
                    optimizer.step()

            avgloss = np.average(loss_vals, weights=batch_sizes)
            if self.verbose >= 2:
                print("Finished epoch", self.epoch_count,
                      "with batch size", batch_size, "and",
                      ("train" if is_train else "validation"),
                      "loss", avgloss, flush=True)

            return avgloss

    def score(self, x_test, y_test):
        if len(y_test.shape) == 1:
            y_test = y_test[:, None]

        if self.scale_data:
            x_test = self.scaler.transform(x_test)

        with torch.no_grad():
            self.neural_net.eval()
            inputv = _np_to_tensor(np.ascontiguousarray(x_test))
            y_test = np.ascontiguousarray(y_test)

            if self.n_classification_labels:
                target = np.array(y_test, dtype=np.int64)
                target = torch.as_tensor(target)
            else:
                target = _np_to_tensor(y_test)

            batch_size = min(self.batch_test_size, x_test.shape[0])

            loss_vals = []
            batch_sizes = []

            tdataset = data.TensorDataset(inputv, target)
            data_loader = data.DataLoader(tdataset,
                batch_size=batch_size, shuffle=True, drop_last=False,
                pin_memory=self.gpu,
                num_workers=self.dataloader_workers,)

            for inputv_this, target_this in data_loader:
                if self.gpu:
                    inputv_this = inputv_this.cuda(non_blocking=True)
                    target_this = target_this.cuda(non_blocking=True)

                batch_actual_size = inputv_this.shape[0]
                theta0v, theta0f, thetas = self.neural_net(inputv_this)

                inputs_ext = inputv_this
                criterion = nn.MSELoss()
                if self.n_classification_labels:
                    inputs_ext = inputv_this[..., None]
                    criterion = nn.CrossEntropyLoss()

                output = (thetas * inputs_ext).sum(1, True)

                if theta0v is not None:
                    output = output + theta0v
                if theta0f is not None:
                    output = output + theta0f

                if self.n_classification_labels:
                    output = output.transpose(1,2)

                loss = criterion(output, target_this)

                loss_vals.append(loss.data.item())
                batch_sizes.append(batch_actual_size)

            return -1 * np.average(loss_vals, weights=batch_sizes)

    def predict_proba(self, x_pred, grad_out=False):
        """
        Predict probabilities of y (if in classification mode).

        Parameters
        ----------
        x_pred : array
            Matrix of features
        grad_out :
            If True, then will output a tuple where the first element is
            the predicted value of y, the second is a numpy array with
            squared gradients of regarding the thetas of each variable,
            and the third is squared gradients of regarding the thetas
            of the varying theta0 (None if self.varying_theta0 is
            False). Note that in case of the second element of the tuple
            the array has shape (no_instances, no_features, no_features)
            where the second dimension refers to denominator of the
            derivative and the third dimension refers to the numerator
            of the derivative). You can concatenate both with:
            `np.concatenate((res[2][:,:,None], res[1]), 2)`
        """
        return self._predict_all(x_pred, grad_out, True)

    def predict(self, x_pred, grad_out=False):
        """
        Predict y.

        Parameters
        ----------
        x_pred : array
            Matrix of features
        grad_out :
            If True, then will output a tuple where the first element is
            the predicted value of y, the second is a numpy array with
            squared gradients of regarding the thetas of each variable,
            and the third is squared gradients of regarding the thetas
            of the varying theta0 (None if self.varying_theta0 is
            False). Note that the second element of the tuple is an
            array of shape (no_instances, no_features, no_features)
            where the second dimension refers to denominator of the
            derivative and the third dimension refers to the numerator
            of the derivative). You can concatenate both with:
            `np.concatenate((res[2][:,:,None], res[1]), 2)`
        """
        return self._predict_all(x_pred, grad_out, False)

    def _predict_all(self, x_pred, grad_out, out_probs):
        if self.scale_data:
            x_pred = self.scaler.transform(x_pred)

        with torch.autograd.set_grad_enabled(grad_out):
            self.neural_net.eval()
            inputv = _np_to_tensor(x_pred)

            if self.gpu:
                inputv = inputv.cuda()

            if grad_out:
                inputv = inputv.requires_grad_(True)

            theta0v, theta0f, thetas = self.neural_net(inputv)

            inputs_ext = inputv
            if self.n_classification_labels:
                inputs_ext = inputv[..., None]

            output_pred = (thetas * inputs_ext).sum(1, True)
            if theta0v is not None:
                output_pred = output_pred + theta0v
            if theta0f is not None:
                output_pred = output_pred + theta0f

            if self.n_classification_labels:
                output_pred = output_pred[:, 0]
                if out_probs:
                    output_pred = F.softmax(output_pred, 1)
                else:
                    output_pred = (torch.max(output_pred, 1,
                        True).indices)

            output_pred = output_pred.data.cpu().numpy()

            # Derivative penalization start
            if grad_out:
                for i in range(thetas.shape[1]):
                    grads_this, = torch.autograd.grad(
                        thetas[:, i].sum(), inputv,
                        retain_graph=True,
                        )

                    grads_this = grads_this[:, :, None].cpu()
                    if i:
                        grads1 = torch.cat((grads1, grads_this), 2)
                    else:
                        grads1 = grads_this

                grads1 = grads1 ** 2.
                grads1 = grads1.numpy()

                if theta0v is not None:
                    grads2, = torch.autograd.grad(
                            theta0v.sum(), inputv,
                            )

                    grads2 = grads2 ** 2.
                    grads2 = grads2.cpu().numpy()
                else:
                    grads2 = None

                return output_pred, grads1, grads2
            # Derivative penalization end

            return output_pred

    def _construct_neural_net(self):
        class NeuralNet(nn.Module):
            def __init__(self, x_dim, y_dim, num_layers,
                         hidden_size, n_classification_labels,
                         varying_theta0, fixed_theta0):
                super(NeuralNet, self).__init__()

                output_hl_size = int(hidden_size)
                self.dropl = nn.Dropout(p=0.5)
                self.varying_theta0 = varying_theta0
                self.fixed_theta0 = fixed_theta0
                self.n_classification_labels = n_classification_labels
                next_input_l_size = x_dim

                if self.fixed_theta0:
                    self.theta0f = nn.Parameter(torch.FloatTensor([.0]))

                llayers = []
                normllayers = []
                for i in range(num_layers):
                    llayers.append(nn.Linear(next_input_l_size,
                                             output_hl_size))
                    normllayers.append(nn.BatchNorm1d(output_hl_size))
                    next_input_l_size = output_hl_size
                    self._initialize_layer(llayers[i])

                self.llayers = nn.ModuleList(llayers)
                self.normllayers = nn.ModuleList(normllayers)

                flo_dim = x_dim + varying_theta0
                if self.n_classification_labels:
                    flo_dim = flo_dim * self.n_classification_labels
                self.fc_last = nn.Linear(next_input_l_size,
                    flo_dim)
                self._initialize_layer(self.fc_last)
                self.num_layers = num_layers
                self.elu = nn.ELU()

            def forward(self, x):
                for i in range(self.num_layers):
                    fc = self.llayers[i]
                    fcn = self.normllayers[i]
                    x = fcn(self.elu(fc(x)))
                    x = self.dropl(x)

                thetas = self.fc_last(x)
                if self.n_classification_labels:
                    thetas = thetas.view(
                        thetas.shape[0],
                        -1,
                        self.n_classification_labels
                        )
                if self.varying_theta0:
                    theta0v = thetas[:, :1]
                    thetas = thetas[:, 1:]
                else:
                    theta0v = None
                    thetas = thetas

                if self.fixed_theta0:
                    theta0f = self.theta0f
                else:
                    theta0f = None

                return theta0v, theta0f, thetas

            def _initialize_layer(self, layer):
                nn.init.constant_(layer.bias, 0)
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(layer.weight, gain=gain)

        self.neural_net = NeuralNet(self.x_dim, self.y_dim,
                                    self.num_layers, self.hidden_size,
                                    self.n_classification_labels,
                                    self.varying_theta0,
                                    self.fixed_theta0)

    def __getstate__(self):
        d = self.__dict__.copy()
        if hasattr(self, "neural_net"):
            state_dict = self.neural_net.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].cpu()
            d["neural_net_params"] = state_dict
            del(d["neural_net"])

        return d

    def __setstate__(self, d):
        self.__dict__ = d

        # backward-compatibility
        if "n_classification_labels" not in d.keys():
            self.n_classification_labels = 0

        if "neural_net_params" in d.keys():
            self._construct_neural_net()
            self.neural_net.load_state_dict(self.neural_net_params)
            del(self.neural_net_params)
            if self.gpu:
                if torch.cuda.is_available():
                    self.move_to_gpu()
                else:
                    self.gpu = False
                    print("Warning: GPU was used to train this model, "
                          "but is not currently available and will "
                          "be disabled "
                          "(renable with estimator move_to_gpu)")

    def get_thetas(self, x_pred, net_scale):
        if self.scale_data:
            x_pred = self.scaler.transform(x_pred)

        with torch.no_grad():
            self.neural_net.eval()
            inputv = _np_to_tensor(x_pred)

            if self.gpu:
                inputv = inputv.cuda()

            theta0v, theta0f, thetas = self.neural_net(inputv)

            if net_scale:
                if theta0v is not None:
                    theta0v = theta0v.data.cpu().numpy()
                if theta0f is not None:
                    theta0f = theta0f.data.cpu().numpy()
                thetas = thetas.data.cpu().numpy()
                return theta0v, theta0f, thetas
            elif not self.scale_data:
                raise ValueError('Must set scale_data to True' +
                'since you did not scale your data')
            else:
                scale = _np_to_tensor(self.scaler.scale_)
                mean = _np_to_tensor(self.scaler.mean_)

                if self.gpu:
                    scale = scale.cuda()
                    mean = mean.cuda()
                if self.n_classification_labels:
                    scale = scale[:, None]
                    mean = mean[:, None]

                thetas = thetas / scale

                if theta0v is None:
                    theta0v = 0
                theta0v = - (mean * thetas).sum(1, True) + theta0v

                theta0v = theta0v.data.cpu().numpy()
                if theta0f is not None:
                    theta0f = theta0f.data.cpu().numpy()
                thetas = thetas.data.cpu().numpy()

                return theta0v, theta0f, thetas


class LLS(BaseEstimator):
    """
    Fit a local linear smoother using a gaussian kernel
    Parameters
    ----------
    kernel_var : float
        variation parameter for the gaussian kernel.
    """
    def __init__(self, kernel_var=1):
        for prop in dir():
            if prop != "self":
                setattr(self, prop, locals()[prop])

    def fit(self, x_train, y_train):

        self.x_train = np.hstack((np.ones((len(x_train), 1)), x_train))
        self.y_train = y_train

        return self

    def predict(self, x, save_thetas=False):

        x = np.hstack((np.ones((len(x), 1)), x))
        pred = np.empty(len(x))
        ed = np.exp(- euclidean_distances(self.x_train, x) / self.kernel_var)
        if save_thetas:
            thetas = np.empty((len(x), x.shape[1]))
        for i in range(x.shape[0]):
            par = self.__get_thetas_single(ed[:, i])
            pred[i] = np.inner(x[i], par)
            if save_thetas:
                thetas[i, :] = par
        if not save_thetas:
            return pred

        return pred, thetas

    def __get_thetas_single(self, k):

        A = self.x_train * np.sqrt(k[:, np.newaxis])
        B = self.y_train * np.sqrt(k)

        thetas = np.linalg.lstsq(A, B, rcond=None)[0]

        return thetas

    def score(self, x, y):

        preds = self.predict(x, save_thetas=False)
        score = np.mean(np.square(y - preds))

        return score
