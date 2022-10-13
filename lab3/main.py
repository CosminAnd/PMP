import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    cutremur = pm.Bernoulli('C', 0.005)
    incendiu_i = pm.Deterministic('I_i', pm.math.switch(cutremur, 0.005, 0.03))
    incendiu = pm.Bernoulli('I', p=incendiu_i)
    alarma_i = pm.Deterministic('A_i', pm.math.switch(cutremur,
                                                  pm.math.switch(incendiu, 0.98, 0.02),
                                                  pm.math.switch(incendiu, 0.95, 0.001)))

    alarma = pm.Bernoulli('A', p=alarma_i)
    trace = pm.sample(20000, chains=1)

    dictionary = {
        'cutremur': trace['C'].tolist(),
        'incendiu': trace['I'].tolist(),
        'alarma': trace['A'].tolist()
    }
    df = pd.DataFrame(dictionary)

    p_cutremur = df[((df['cutremur'] == 1) & (df['alarma'] == 1))].shape[0] / df[df['cutremur'] == 1].shape[0]
    print(p_cutremur)


    p_incendiu = df[((df['incendiu'] == 1) & (df['alarma'] == 0))].shape[0] / df[df['alarma'] == 0].shape[0]
    print(p_incendiu)

    az.plot_posterior(trace)
    plt.show()