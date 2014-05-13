#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import math
from emcee import EnsembleSampler
from emcee.utils import sample_ball

class LnPost(object):
    """
    Class that represents log of posterior density for normal model.
    """
    def __init__(self, yij, lnpr=None):
        self._lnlike = LnLike(yij)
        self._lnpr = lnpr

    def __call__(self, p):
        return self._lnlike(p) + self._lnpr(p)


class LnLike(object):
    """
    Class that represents ln of likelihood for normal model.
    """

    @staticmethod
    def ln_of_normpdf(x, loc=0., logscale=0.):
        """
        Returns ln of pdf for normal distribution at points x.

        :param x:
            Value (or array of values) for which to calculate the result.

        :param loc (optional):
            Mean of the distribution.

        :param logscale (optional):
            Log of std of the distribution.

        :return:
            Log of pdf for normal distribution with specified parameters. Value
            or array of values (depends on ``x``).
        """
        return -0.5 * math.log(2. * math.pi) - logscale -\
               (x - loc) ** 2 / (2. * math.exp(logscale) ** 2)

    def __init__(self, y_ij):
        self.y_ij = y_ij

    def __call__(self, p):
        """
        Returns ln of likelihood for normal model.

        :param p:
            Parameters of model. [mu, log_sigma, log_tau, theta_j j=1,...,N]

        :return:
            Log of likelihood function.
        """
        k = sum([np.sum(self.ln_of_normpdf(self.y_ij[j], loc=theta,
                                           logscale=p[2])) for j, theta in
                 enumerate(p[3:])])
        return np.sum(self.ln_of_normpdf(p[3:], loc=p[0], logscale=p[1])) + k

def lnpr(p):
    """
    Prior on parameters.

    :param p:
        Parameters of model. [mu, log_sigma, log_tau, theta_j j=1,...,N]

    :return:
        Log of prior density for given parameter ``p``.
    """

    return p[2]


if __name__ == '__main__':
    # Data from Gelman (Table 11.2)
    y_ij = list()
    y_ij.append([62, 60, 63, 59])
    y_ij.append([63, 67, 71, 64, 65, 66])
    y_ij.append([68, 66, 71, 67, 68, 68])
    y_ij.append([56, 62, 60, 61, 63, 64, 63, 59])
    print y_ij
    lnpost = LnPost(y_ij, lnpr=lnpr)
    ndim = 7
    nwalkers = 500
    sampler = EnsembleSampler(nwalkers, ndim, lnpost)
    p0 = sample_ball([60., 0., 0., 60., 60., 60., 60.],
                     [10., 1., 1., 10., 10., 10., 10.], size=500)
    print "sample ball :"
    print p0
    print "Burning-in..."
    pos, prob, state = sampler.run_mcmc(p0, 300)
    print "Reseting burn-in..."
    sampler.reset()
    print "Now sampling from posterior"
    sampler.run_mcmc(pos, 1000, rstate0=state)
