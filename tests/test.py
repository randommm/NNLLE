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
#along with this program.    If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
from nnlocallinear import NLS, NNPredict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from generate_data import generate_data
import pandas as pd
import itertools
import os

test_classification = True
test_nnpred = False

s_penalization_thetas = [0.0, 0.01, 0.1, 0.5]
s_penalization_variable_theta0 = [0.0, 0.01, 0.1, 0.5]
s_scale_data = [True, False]
s_complexity = [10, 100, 1000]
s_varying_theta0 = [True, False]
s_fixed_theta0 = [True, False]

prods = itertools.product(s_penalization_thetas,
    s_penalization_variable_theta0, s_scale_data, s_complexity,
    s_varying_theta0, s_fixed_theta0)
prods = list(prods)
prods = np.random.permutation(np.array(prods, dtype=object))

df = pd.DataFrame(columns=[
    'penalization_thetas', 'penalization_variable_theta0',
    'scale_data', 'complexity',
    'varying_theta0', 'fixed_theta0', 'mse_val', 'mse_test',

])

for penalization_thetas, penalization_variable_theta0, scale_data, complexity, varying_theta0, fixed_theta0 in prods:
    np.random.seed(10)

    n_train = 1_000
    n_val = 1_000
    n_test = 5_000
    x_train, y_train = generate_data(n_train)
    x_val, y_val = generate_data(n_val)
    x_test, y_test = generate_data(n_test)

    if test_classification:
        cutp1 = (max(y_train) + min(y_train)) / 3
        cutp2 = 2*(max(y_train) + min(y_train)) / 3

        y_train_n = np.zeros_like(y_train)
        y_train_n[y_train>cutp1] = 1
        y_train_n[y_train>cutp2] = 2
        y_train = y_train_n

        y_val_n = np.zeros_like(y_val)
        y_val_n[y_val>cutp1] = 1
        y_val_n[y_val>cutp2] = 2
        y_val = y_val_n

        y_test_n = np.zeros_like(y_test)
        y_test_n[y_test>cutp1] = 1
        y_test_n[y_test>cutp2] = 2
        y_test = y_test_n
        n_classification_labels = 3
    else:
        n_classification_labels = 0

    np.random.seed()

    # print(y_train)
    # print(min(y_train))
    # print(max(y_train))

    params = dict(
    es_give_up_after_nepochs=1,
    verbose=2,
    es=True,
    hidden_size=complexity,
    num_layers=3,
    gpu=True,
    dataloader_workers=0,
    batch_initial=100,
    n_classification_labels = n_classification_labels,
    )

    extra_param = dict(
    scale_data=scale_data,
    varying_theta0=varying_theta0,
    fixed_theta0=fixed_theta0,

    penalization_thetas=penalization_thetas,
    penalization_variable_theta0= penalization_variable_theta0,
    )

    if test_nnpred:
        nnlocallinear_obj = NNPredict(**params)
    else:
        nnlocallinear_obj = NLS(**params, **extra_param)

    nnlocallinear_obj.fit(x_train, y_train)

    nnlocallinear_obj.verbose = 0
    mse_val = - nnlocallinear_obj.score(x_val, y_val)
    print("mse on val (locallinearr):", mse_val)
    print("mse on val (locallinearr):",
          ((nnlocallinear_obj.predict(x_val) - y_val)**2).mean()
         )

    mse_test = - nnlocallinear_obj.score(x_test, y_test)
    print("mse on test (locallinearr):", mse_test)
    print("mse on test (locallinearr):",
          ((nnlocallinear_obj.predict(x_test) - y_test)**2).mean()
         )

    if test_classification:
        pred_prob = nnlocallinear_obj.predict_proba(x_test)
        print("test predict_proba (locallinearr):",
            np.isclose(pred_prob.sum(1), 1).all(),
            (pred_prob >= 0).all(),
            (pred_prob <= 1).all(),
            )

    print("predict on test (locallinearr):",
          (nnlocallinear_obj.predict(x_test)).mean()
         )


    if not test_nnpred:
        preds, grads1, grads2 = nnlocallinear_obj.predict(x_test, True)
        print("predict on test with grad (locallinearr):",
              preds.mean(), " ", grads1.mean(),
              (grads2 if grads2 is not None else None)
             )

        if scale_data:
            print(nnlocallinear_obj.get_thetas(x_test, False)[:2])
        print(nnlocallinear_obj.get_thetas(x_test, True)[:2])

    df.loc[len(df)] = (penalization_thetas,
        penalization_variable_theta0, scale_data, complexity,
        varying_theta0, fixed_theta0, mse_val, mse_test)
    df = df.sort_values(list(df.columns))
    print(df)
    df.to_csv('results.csv')
