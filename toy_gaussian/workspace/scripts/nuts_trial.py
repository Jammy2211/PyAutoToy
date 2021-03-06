import numpy as np

from autofit.optimize.non_linear.nuts import nuts6


def correlated_normal(theta):
    """ Example of a target distribution that could be sampled from using NUTS.  (Doesn't include the normalizing constant.)
    Note:
    cov = np.asarray([[1, 1.98],
                      [1.98, 4]])
    """

    # A = np.linalg.inv( cov )
    A = np.asarray([[50.251256, -24.874372], [-24.874372, 12.562814]])

    grad = -np.dot(theta, A)
    logp = 0.5 * np.dot(grad, theta.T)
    return logp, grad


D = 2
M = 5000
Madapt = 5000
theta0 = np.random.normal(0, 1, D)
delta = 0.2

mean = np.zeros(2)
cov = np.asarray([[1, 1.98], [1.98, 4]])

samples, lnprob, epsilon = nuts6(correlated_normal, M, Madapt, theta0, delta)

samples = samples[1::10, :]
print("Mean: {}".format(np.mean(samples, axis=0)))
print("Stddev: {}".format(np.std(samples, axis=0)))

import pylab as plt

temp = np.random.multivariate_normal(mean, cov, size=500)
plt.plot(temp[:, 0], temp[:, 1], ".")
plt.plot(samples[:, 0], samples[:, 1], "r+")
plt.show()
