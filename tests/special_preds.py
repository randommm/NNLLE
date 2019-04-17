import numpy as np
import pandas as pd
from nnlocallinear import NNPredict, LLE
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

x = np.linspace(0, 100, 2000)
np.random.seed(0)
y = x + 20 * np.sin(x/10) + np.random.normal(0, 3, 2000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=10,
    num_layers=1,
    gpu=False,
    tuningp=0,
    scale_data=True,
    dataloader_workers=0).fit(x_train.reshape(-1, 1), y_train)

pred0 = model.predict(x_test[0:5].reshape(-1, 1))
pred1 = model._special_predict(x_test[0:5].reshape(-1, 1))
pred2 = model._special_predict2(x_test[0:5].reshape(-1, 1))

print('pred normal:', pred0.tolist())
print('pred special 1:', pred1.tolist())
print('pred special 2:', pred2.tolist())