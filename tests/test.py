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

s_tuningp = [0.0, 0.01, 0.1, 0.5]
s_scale_data = [True, False]
s_complexity = [10, 100, 1000]

prods = itertools.product(s_tuningp, s_scale_data, s_complexity)
prods = list(prods)

df = pd.DataFrame(columns=[
    'tuningp', 'scale_data', 'complexity', 'mse_val', 'mse_test'
])

for tuningp, scale_data, complexity in prods:
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
    verbose=1,
    es=True,
    hidden_size=complexity,
    num_layers=3,
    gpu=True,
    dataloader_workers=1,
    tuningp=tuningp,
    scale_data=scale_data,
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
    # print(nnlocallinear_obj.get_thetas(x_test, True))
    # print(nnlocallinear_obj.get_thetas(x_test, False))

    df.loc[len(df)] = (tuningp, scale_data, complexity, mse_val,
        mse_test)
    print(df)
    df.to_csv('results.csv')