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
from nnlocallinear import NNPredict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from generate_data import generate_data
import pandas as pd
import itertools

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

    n_train = 9_000
    n_val = 1_000
    n_test = 5_000
    x_train, y_train = generate_data(n_train)
    x_val, y_val = generate_data(n_val)
    x_test, y_test = generate_data(n_test)

    np.random.seed()
    # print(y_train)
    # print(min(y_train))
    # print(max(y_train))

    nnlocallinear_obj = NNPredict(
    es_give_up_after_nepochs=30,
    verbose=1,
    es=True,
    hidden_size=complexity,
    num_layers=3,
    gpu=True,
    dataloader_workers=1,

    scale_data=scale_data,
    varying_theta0=varying_theta0,
    fixed_theta0=fixed_theta0,

    penalization_thetas=penalization_thetas,
    penalization_variable_theta0= penalization_variable_theta0,
    )
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

    print("predict on test (locallinearr):",
          (nnlocallinear_obj.predict(x_test)).mean()
         )
    preds, grads = nnlocallinear_obj.predict(x_test, True)
    print("predict on test with grad (locallinearr):",
          preds.mean(), " ", grads.mean(),
         )
    # print("special predict on test (locallinearr):",
          # (nnlocallinear_obj._special_predict(x_test)).mean()
         # )
    # print("special predict 2 on test (locallinearr):",
          # (nnlocallinear_obj._special_predict2(x_test)).mean()
         # )

    if scale_data:
        print(nnlocallinear_obj.get_thetas(x_test, False)[:2])
    print(nnlocallinear_obj.get_thetas(x_test, True)[:2])

    df.loc[len(df)] = (penalization_thetas,
        penalization_variable_theta0, scale_data, complexity,
        varying_theta0, fixed_theta0, mse_val, mse_test)
    df = df.sort_values(list(df.columns))
    print(df)
    df.to_csv('results.csv')
