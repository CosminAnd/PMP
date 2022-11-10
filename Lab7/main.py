import numpy as np
import matplotlib as plt
import arviz as az
import pandas as pd
import pymc3 as pm

data = pd.read_csv('Prices.csv')
data['logHD'] = np.log(data['HardDrive'])
data['logHD'] = data['logHD'].replace([np.inf, -np.inf], np.nan)
data = data.dropna()
data = data.reset_index(drop=True)

#Ex1
with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    mu = alpha+beta1*data['Speed']+ beta2* data['logHD']
    y = pm.Normal('y', mu=mu, sigma=sigma, observed= data['Price'])
    trace = pm.sample(500, tune=100, chains=1)