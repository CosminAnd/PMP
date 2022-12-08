import pymc3 as pm
import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
#import theano
#theano.config.blas__ldflags = ''

# Ex1
if __name__ == '__main__':
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date.csv')
    """
    #Ex 2 - generare 500 de date
    dummy_data = np.loadtxt('date500.csv')
    random.shuffle(dummy_data)
    """

    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 3
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    with pm.Model() as model_p:
        alpha = pm.Normal('α', mu=0, sd=1)
        beta = pm.Normal('β', mu=0, sd=1, shape=order)
        # punctul b
        #beta = pm.Normal('β', mu=0, sd=100, shape=order)
        #beta = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        eps = pm.HalfNormal('ε', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=miu, sd=eps, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()

    plt.show()



