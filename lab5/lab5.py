"""
Gaussian Mixture of Models
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn


def em_gmm_orig(xs, pis, mus, sigmas, tol=0.01, max_iter=10):

    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for i in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(len(mus)):
            for i in range(n):
                ws[j, i] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
        ws /= ws.sum(0)

        # M-step
        pis = np.zeros(k)
        for j in range(len(mus)):
            for i in range(n):
                pis[j] += ws[j, i]
        pis /= n

        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += ws[j, i] * xs[i]
            mus[j] /= ws[j, :].sum()

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i]- mus[j], (2,1))
                sigmas[j] += ws[j, i] * np.dot(ys, ys.T)
            sigmas[j] /= ws[j,:].sum()

    return pis, mus, sigmas

pis = np.random.random(6)
pis /= pis.sum()
mus = np.random.random((6,2))
sigmas = np.array([np.eye(2)] * 6)
ll1, pis1, mus1, sigmas1 = em_gmm_orig(X_2, pis, mus, sigmas)

print(mus1)
print(sigmas1)

