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

from nnlocallinear import NNPredict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from generate_data import generate_data

if __name__ == '__main__':
    n_train = 10_000
    n_test = 5_000
    x_train, y_train = generate_data(n_train)
    x_test, y_test = generate_data(n_test)

    print(y_train)
    print(min(y_train))
    print(max(y_train))

    nnlocallinear_obj = NNPredict(
    verbose=2,
    es=True,
    hidden_size=100,
    num_layers=3,
    gpu=True,
    dataloader_workers=0,
    tuningp=0.5,
    )
    nnlocallinear_obj.fit(x_train, y_train)

    nnlocallinear_obj.verbose = 0
    print("Risk on test (locallinearr):", - nnlocallinear_obj.score(x_test, y_test))
    print("Risk on test (locallinearr):",
          ((nnlocallinear_obj.predict(x_test) - y_test)**2).mean()
         )
    print(nnlocallinear_obj.get_thetas(x_test, True))
    print(nnlocallinear_obj.get_thetas(x_test, False))


    nnlocallinear_obj = NNPredict(
    verbose=2,
    es=True,
    hidden_size=100,
    num_layers=3,
    gpu=True,
    dataloader_workers=0,
    tuningp=0.5,
    scale_data=False,
    )
    nnlocallinear_obj.fit(x_train, y_train)

    nnlocallinear_obj.verbose = 0
    print("Risk on test (locallinearr):", - nnlocallinear_obj.score(x_test, y_test))
    print("Risk on test (locallinearr):",
          ((nnlocallinear_obj.predict(x_test) - y_test)**2).mean()
         )
    print(nnlocallinear_obj.get_thetas(x_test, True))
