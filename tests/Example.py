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

def func(hidden_size, num_layers, fig):

    output = pd.DataFrame(index=['No penalty', 'Low penalty', 'High penalty'], columns=['MSE', 'MAE', 'SD'])

    model = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=hidden_size,
        num_layers=num_layers,
        gpu=False,
        tuningp=0,
        scale_data=True,
        dataloader_workers=0).fit(x_train.reshape(-1, 1), y_train)

    model2 = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=hidden_size,
        num_layers=num_layers,
        gpu=False,
        tuningp=50,
        scale_data=True,
        dataloader_workers=0).fit(x_train.reshape(-1, 1), y_train)

    model3 = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=hidden_size,
        num_layers=num_layers,
        gpu=False,
        tuningp=1000,
        scale_data=True,
        dataloader_workers=0).fit(x_train.reshape(-1, 1), y_train)

    f, ax = plt.subplots(ncols=3)
    ax[0].plot(x, y, 'black', linewidth=4, label='True regression')
    ax[0].plot(x_test, model.predict(x_test.reshape(-1, 1)), 'go', linewidth=4, label='No penalty')
    ax[0].plot(x_test, model2.predict(x_test.reshape(-1, 1)), 'bo', linewidth=4, label='Low penalty')
    ax[0].plot(x_test, model3.predict(x_test.reshape(-1, 1)), 'ro', linewidth=4, label='High penalty')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend(loc=2)

    ax[1].plot(x_test, model.get_thetas(x_test.reshape(-1, 1))[:, 0].tolist(), 'go', linewidth=4, label='No penalty')
    ax[1].plot(x_test, model2.get_thetas(x_test.reshape(-1, 1))[:, 0].tolist(), 'bo', linewidth=4, label='Low penalty')
    ax[1].plot(x_test, model3.get_thetas(x_test.reshape(-1, 1))[:, 0].tolist(), 'ro', linewidth=4, label='High penalty')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('theta_0')
    ax[1].legend(loc=2)

    ax[2].plot(x_test, model.get_thetas(x_test.reshape(-1, 1))[:, 1].tolist(), 'go', linewidth=4, label='No penalty')
    ax[2].plot(x_test, model2.get_thetas(x_test.reshape(-1, 1))[:, 1].tolist(), 'bo', linewidth=4, label='Low penalty')
    ax[2].plot(x_test, model3.get_thetas(x_test.reshape(-1, 1))[:, 1].tolist(), 'ro', linewidth=4, label='High penalty')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('theta_1')
    ax[2].legend(loc=2)
    f.savefig(fig, bbox_inches='tight')

    pred = model.predict(x_test.reshape(-1, 1))
    output.loc['No penalty'] = [mse(y_test, pred), mae(y_test, pred), np.std(y_test-pred)]
    pred = model2.predict(x_test.reshape(-1, 1))
    output.loc['Low penalty'] = [mse(y_test, pred), mae(y_test, pred), np.std(y_test-pred)]
    pred = model3.predict(x_test.reshape(-1, 1))
    output.loc['High penalty'] = [mse(y_test, pred), mae(y_test, pred), np.std(y_test-pred)]

    return output


print(func(50, 1, 'plots/1_50.pdf'))
print(func(100, 1, 'plots/1_100.pdf'))
print(func(200, 1, 'plots/1_200.pdf'))
print(func(500, 1, 'plots/1_500.pdf'))
print(func(50, 3, 'plots/3_50.pdf'))
print(func(100, 3, 'plots/3_100.pdf'))
print(func(200, 3, 'plots/3_200.pdf'))
print(func(500, 3, 'plots/3_500.pdf'))
print(func(50, 5, 'plots/5_50.pdf'))
print(func(200, 5, 'plots/5_100.pdf'))
print(func(300, 5, 'plots/5_200.pdf'))
print(func(500, 5, 'plots/5_500.pdf'))

model4 = LLE().fit(x_train.reshape(-1, 1), y_train)
pred, thetas = model4.predict(x_test.reshape(-1, 1), 1, True)

f, ax = plt.subplots(ncols=3)
ax[0].plot(x, y, 'ko', linewidth=4, label='True regression')
ax[0].plot(x_test, pred, 'go', linewidth=4, label='LLE')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].legend(loc=2)

ax[1].plot(x_test, thetas[:, 0], 'go', linewidth=4)
ax[1].set_xlabel('x')
ax[1].set_ylabel('theta_0')

ax[2].plot(x_test, thetas[:, 1], 'go', linewidth=4)
ax[2].set_xlabel('x')
ax[2].set_ylabel('theta_1')
f.savefig('plots/LL.pdf', bbox_inches='tight')

print([mse(y_test, pred), mae(y_test, pred), np.std(y_test - pred)])

model = NNPredict(
    verbose=0,
    es=True,
    es_give_up_after_nepochs=50,
    hidden_size=500,
    num_layers=5,
    gpu=False,
    tuningp=0,
    scale_data=True,
    dataloader_workers=0).fit(x_train.reshape(-1, 1), y_train)

f, ax = plt.subplots(ncols=2)
ax[0].plot(x, y, 'ko', linewidth=4, label='True regression')
ax[0].plot(x_test, model.predict(x_test.reshape(-1, 1)), 'go', linewidth=4, label='No penalty')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].legend(loc=2)

ax[1].plot(x, y, 'ko', linewidth=4, label='True regression')
ax[1].plot(x_test, pred, 'go', linewidth=4, label='LLE')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].legend(loc=2)
f.savefig('plots/comparison.pdf', bbox_inches='tight')
