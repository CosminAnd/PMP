import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az


if __name__ == '__main__':
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 2
    order2 = 3
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1c = np.vstack([x_1 ** i for i in range(1, order2 + 1)])
    x_1cs = (x_1c - x_1c.mean(axis=1, keepdims=True)) / x_1c.std(axis=1, keepdims=True)
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_p:
        alpha = pm.Normal('α', mu=0, sd=1)
        beta = pm.Normal('β', mu=0, sd=10, shape=order)
        eps = pm.HalfNormal('ε', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=miu, sd=eps, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_c:
        alpha1 = pm.Normal('α1', mu=0, sd=1)
        beta1 = pm.Normal('β1', mu=0, sd=10, shape=order2)
        eps1 = pm.HalfNormal('ε1', 5)
        miu1 = alpha1 + pm.math.dot(beta1, x_1cs)
        y_pred1 = pm.Normal('y_pred1', mu=miu, sd=eps, observed=y_1s)
        idata_c = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    # linear
    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
    β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
    y_l_post = α_l_post + β_l_post * x_new
    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    # patratic
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

    # cubic
    α_c_post = idata_c.posterior['α1'].mean(("chain", "draw")).values
    β_c_post = idata_c.posterior['β1'].mean(("chain", "draw")).values
    idx = np.argsort(x_1cs[0])
    y_c_post = α_c_post + np.dot(β_c_post, x_1cs)
    plt.plot(x_1cs[0][idx], y_c_post[idx], 'C3', label=f'model order2 {order2}')

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()

    #plt.show()

    # Punctul 3
    waic_l = az.waic(idata_l, scale="deviance")
    loo_l = az.loo(idata_l, scale="deviance")
    print(waic_l)
    print(loo_l)
    cmp_df = az.compare({'model_l': idata_l, 'model_p': idata_p, 'model_c':idata_c},
                         method='BB-pseudo-BMA', ic="waic", scale="deviance")
    az.plot_compare(cmp_df)
    plt.show()
