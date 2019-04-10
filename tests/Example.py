import numpy as np
from nnlocallinear import NNPredict, LLE
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

x = np.linspace(0, 100, 2000)
np.random.seed(0)
y = x + 20 * np.sin(x/10) + np.random.normal(0, 3, 2000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def func(hidden_size, num_layers, fig):

    model = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=hidden_size,
        num_layers=num_layers,
        gpu=False,
        tuningp=0,
        dataloader_workers=0).fit(x_train.reshape(-1, 1), y_train)

    model2 = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=hidden_size,
        num_layers=num_layers,
        gpu=False,
        tuningp=50,
        dataloader_workers=0).fit(x_train.reshape(-1, 1), y_train)

    model3 = NNPredict(
        verbose=0,
        es=True,
        es_give_up_after_nepochs=50,
        hidden_size=hidden_size,
        num_layers=num_layers,
        gpu=False,
        tuningp=1000,
        dataloader_workers=0).fit(x_train.reshape(-1, 1), y_train)

    f, ax = plt.subplots(ncols=3)
    ax[0].plot(x, y, 'black', linewidth=4, label='True regression')
    ax[0].plot(x_test, model.predict(x_test.reshape(-1, 1)), 'go', linewidth=4, label='No penalty')
    ax[0].plot(x_test, model2.predict(x_test.reshape(-1, 1)), 'bo', linewidth=4, label='Low penalty')
    ax[0].plot(x_test, model3.predict(x_test.reshape(-1, 1)), 'ro', linewidth=4, label='High penalty')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend(loc=2)

    ax[1].plot(x_test, model.get_thetas(x_test.reshape(-1, 1), original_scale=True)[:, 0].tolist(), 'go', linewidth=4, label='No penalty')
    ax[1].plot(x_test, model2.get_thetas(x_test.reshape(-1, 1), original_scale=True)[:, 0].tolist(), 'bo', linewidth=4, label='Low penalty')
    ax[1].plot(x_test, model3.get_thetas(x_test.reshape(-1, 1), original_scale=True)[:, 0].tolist(), 'ro', linewidth=4, label='High penalty')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('theta_0')
    ax[1].legend(loc=2)

    ax[2].plot(x_test, model.get_thetas(x_test.reshape(-1, 1), original_scale=True)[:, 1].tolist(), 'go', linewidth=4, label='No penalty')
    ax[2].plot(x_test, model2.get_thetas(x_test.reshape(-1, 1), original_scale=True)[:, 1].tolist(), 'bo', linewidth=4, label='Low penalty')
    ax[2].plot(x_test, model3.get_thetas(x_test.reshape(-1, 1), original_scale=True)[:, 1].tolist(), 'ro', linewidth=4, label='High penalty')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('theta_1')
    ax[2].legend(loc=2)
    f.savefig(fig, bbox_inches='tight')


func(10, 1, 'plots/1_10.pdf')
func(100, 1, 'plots/1_100.pdf')
func(300, 1, 'plots/1_300.pdf')
func(10, 3, 'plots/3_10.pdf')
func(100, 3, 'plots/3_100.pdf')
func(300, 3, 'plots/3_300.pdf')


model4 = LLE().fit(x_train.reshape(-1, 1), y_train)
pred, thetas = model4.predict(x_test.reshape(-1, 1), 1, True)

f, ax = plt.subplots(ncols=3)
ax[0].plot(x, y, 'ko', linewidth=4, label='True regression')
ax[0].plot(x_test, pred, 'go', linewidth=4, label='No penalty')
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