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


class Dlognorm_shifted(object):
    """
    Function that returns samples from shifted lognormal distribution.
    """
    def __init__(self, shift=0):
        self.shift = shift

    def __call__(self, **kwargs):
        return self.shift + np.random.lognormal(**kwargs)

def lnpr(p):
    """
    Prior on parameters.

    :param p:
        Parameters of model. [mu, log_sigma, log_tau, theta_j j=1,...,N]

    :return:
        Log of prior density for given parameter ``p``.
    """

    return p[2]

def get_theta_sample(dgamma, a=2, **kwargs):
    """
    Function that generates sample from theta distribution given specified gamma
    distribution.

    :param dgamma:
        Distribution of gamma. When called with **kwargs returns samples from
        distribution of gamma.

    :return:
        Ln of pdf of theta.
    """

    def theta_cdf(theta, beta, a=2):
        """
        Function that returns cdf for theta given value(s) of beta.
        """
        return ((1. - beta) ** (-a) - (1. - beta * math.cos(theta)) ** (-a)) /\
               ((1. - beta) ** (-a) - 1.)

    # Sample N points from gamma_distr
    gamma_samples = np.asarray(dgamma(**kwargs))
    # Recalculate beta distribution
    beta_samples = np.sqrt(gamma_samples ** 2. - 1.) / gamma_samples

    # Sample N uniform numbers from unif(0, 1)
    us = np.random.uniform(0, 1, size=len(beta_samples))
    # Use inverse CDF method for sampling N values of theta from theta
    # distribution
    theta_samples = np.arccos((1. - ((1. - beta_samples) ** (-a) - us *
                                     ((1. - beta_samples) ** (-a) - 1.)) **
                               (-1. / a)) / beta_samples)

    return theta_samples

# TODO: Calculate theta_pdf for whole distribution of gamma/beta and fit with
# KDE, then, for each object, get theta_pdf_i using importance sampling with
# KDE as importance function


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

    dgamma = Dlognorm_shifted(2.)
    np.hist(dgamma(mean=2, sigma=0.5, size=10000), bins=50, normed=True)
    np.hist(get_theta_sample(dgamma, mean=2, sigma=0.5, size=10000)*180./math.pi, bins=50, normed=True)
