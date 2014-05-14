#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import math


class Dlognorm_shifted(object):
    """
    Function that returns samples from shifted lognormal distribution.
    """
    def __init__(self, shift=0):
        self.shift = shift

    def __call__(self, **kwargs):
        return self.shift + np.random.lognormal(**kwargs)


class LnPost(object):
    """
    Class that represents log of posterior density for kinematic model.
    """
    def __init__(self, beta_ij, lnpr=None):
        self._lnlike = LnLike(beta_ij)
        self._lnpr = lnpr

    def __call__(self, p):
        return self._lnlike(p) + self._lnpr(p)


class LnLike(object):
    """
    Class that represents ln of likelihood for kinematic model.
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

    def __init__(self, beta_ij):
        self._beta_ij = beta_ij

    def __call__(self, p):
        """
        Returns ln of likelihood for kinematic model.

        :param p:
            Parameters of model. [mu_{G}^{Gamma}, ln(tau_{G}^{Gamma}),
            mu_{j}^{Gamma}, ln(tau_{j}_{Gamma}), j=1,...,N_{sources}]

        :return:
            Ln of likelihood function.
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
        return ((1. - beta) ** (-a) - (1. - beta * math.cos(theta)) ** (-a)) / \
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
    dgamma = Dlognorm_shifted(2.)
    np.hist(dgamma(mean=2, sigma=0.5, size=10000), bins=50, normed=True)
    np.hist(get_theta_sample(dgamma, mean=2, sigma=0.5, size=10000) * 180. /
            math.pi, bins=50, normed=True)

    # Load data
    data_lists = list()
    files = ['source.txt', 'beta.txt', 'sigma_beta.txt']
    for file_ in files:
        with open(file_) as f:
            data_lists.append(f.read().splitlines())
    sources_list = data_lists[0]
    betas_list = data_lists[1]
    beta_sigmas_list = data_lists[2]

    # Convert to float numbers
    for i, value in enumerate(betas_list):
        try:
            betas_list[i] = float(value)
        except ValueError:
            betas_list[i] = None

    for i, value in enumerate(beta_sigmas_list):
        try:
            beta_sigmas_list[i] = float(value)
        except ValueError:
            beta_sigmas_list[i] = None

    #as Bring all data to nested list
    betas_nested = list()
    beta_sigmas_nested = list()
    betas_dict = dict()
    beta_sigmas_dict = dict()
    sources_set = set(sources_list)
    for source in sources_set:
        # Indices of ``source_silt`` with current source
        indices = [i for i, x in enumerate(sources_list) if x == source]
        betas_nested.append([betas_list[i] for i in indices])
        betas_dict.update({source: [betas_list[i] for i in indices]})
        beta_sigmas_nested.append([beta_sigmas_list[i] for i in indices])
        beta_sigmas_dict.update({source: [beta_sigmas_list[i] for i in
                                          indices]})



