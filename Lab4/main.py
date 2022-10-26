import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import arviz as az

if __name__ == '__main__':
    model = pm.Model()
    with model:
        alpha = 4.66  #aproximativ, aflat prin incercari; intre 4.6-4.9
        clineti_pe_ora = pm.Poisson('clineti_pe_ora', alpha)
        timp_asteptare = pm.Normal('timp_asteptare', mu=1, sigma=1 / 2)
        comanda = pm.Exponential('comanda', 1 / alpha)
        trace = pm.sample(100)

    dictionary = {
        'timp_preparare': trace['comanda'].tolist(),
        'timp_asteptare': trace['timp_asteptare'].tolist(),
    }
    df = pd.DataFrame(dictionary)

    timp_asteptare15 = df[(df['timp_asteptare'] + df['timp_preparare'] <= 15)]
    print("Timpul de asteptare <= 15:", timp_asteptare15.shape[0] / df.shape[0])

    total = 0
    #timpul de servire pentru un client este timpul asteptare + timpul de preparare
    for tup in zip(df['timp_asteptare'], df['timp_preparare']):
        total += tup[0] + tup[1]

    print("Timpul mediu de asteptare:", total / df.shape[0])


    az.plot_posterior({"Clienti pe ora": trace['clineti_pe_ora'], "Timp de asteptare": trace['timp_asteptare'], "Medie comenzi": trace['comanda']})
    plt.show()
