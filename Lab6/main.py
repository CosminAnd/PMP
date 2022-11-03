import csv
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    ss_xy = np.sum(y * x) - n * m_y * m_x
    ss_xx = np.sum(x * x) - n * m_x * m_x
    b_1 = ss_xy / ss_xx
    b_0 = m_y - b_1 * m_x

    return b_0, b_1


def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m",
                marker="o", s=30)
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="g")
    plt.xlabel('ppvt')
    plt.ylabel('momage')
    plt.show()

def ex1(data):
    np.random.seed(1)
    x = data['momage']
    y = data['ppvt']
    _, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(x, y, 'C0.')
    ax[0].set_xlabel('momage')
    ax[0].set_ylabel('ppvt', rotation=0)
    az.plot_kde(y, ax=ax[1])
    ax[1].set_xlabel('y')
    plt.tight_layout()
    plt.show()

def main():
    data = np.genfromtxt('data.csv', delimiter=',', names=True)
    ex1(data)

    with open("data.csv", 'r') as file:
        csvreader = csv.reader(file)
        ppvt = []
        mom_age = []
        for row in csvreader:
            ppvt.append(row[1])
            mom_age.append(row[3])
        ppvt.remove(ppvt[0])
        mom_age.remove(mom_age[0])
        for i in range(0, len(ppvt)):
            ppvt[i] = int(ppvt[i])
            mom_age[i] = int(mom_age[i])
    ppvt = np.array(ppvt)
    mom_age = np.array(mom_age)

    b = estimate_coef(ppvt, mom_age)
    plot_regression_line(ppvt, mom_age, b)


main()
