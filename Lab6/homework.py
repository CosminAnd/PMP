import csv
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm

if __name__ == '__main__':
    data = np.genfromtxt('data.csv', delimiter=',', names=True)

    with open("data.csv", 'r') as file:
        csvreader = csv.reader(file)
        ppvt = []
        educ_cat = []
        mom_age = []
        for row in csvreader:
            ppvt.append(row[1])
            educ_cat.append(row[2])
            mom_age.append(row[3])
        ppvt.remove(ppvt[0])
        educ_cat.remove(educ_cat[0])
        mom_age.remove(mom_age[0])
        for i in range(0, len(ppvt)):
            ppvt[i] = int(ppvt[i])
            mom_age[i] = int(mom_age[i])
            educ_cat[i] = int(educ_cat[i])
    ppvt = np.array(ppvt)
    mom_age = np.array(mom_age)
    educ_cat = np.array(educ_cat)

    plt.scatter(mom_age, ppvt)
    plt.xlabel('educ_cat')
    plt.ylabel('ppvt')
    plt.title('Mom age')
    #plt.show()

    csv_model = pm.Model()
    with csv_model:
        ppvt_sd = np.std(ppvt)
        ages_sd = np.std(educ_cat)
        alfa = pm.Normal('alfa', mu=0, sd=10 * (ppvt_sd))
        beta = pm.Normal("beta", mu=0, sd=1 * ppvt_sd / ages_sd)
        epsilon = pm.HalfCauchy("epsilon", 5)
        miu = pm.Deterministic('miu', alfa + beta * educ_cat)
        ppvt_pred = pm.Normal('ppvt_pred', mu=miu, sd=epsilon, observed=ppvt)

    idata_g = pm.sample(1000, tune=2300, return_inferencedata=True, model=csv_model, chains= 1)
    az.plot_trace(idata_g, var_names=['alfa', 'beta', 'epsilon'],show=True)


    alpha_m = alfa.mean()
    beta_m = beta.mean()
    ppc = pm.sample_posterior_predictive(idata_g, samples=100, model=csv_model)
    az.plot_trace(educ_cat, ppvt, show=True)
    az.plot_trace(educ_cat, alpha_m + beta_m * educ_cat, show=True)
    az.plot_hdi(educ_cat, ppc['ppvt_pred'], hdi_prob=0.5, color='grappvt', smooth=False, show=True)
    az.plot_hdi(educ_cat, ppc['ppvt_pred'], color='gray', smooth=False, show=True)

