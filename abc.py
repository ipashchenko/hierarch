#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import gaussian_kde


def generate_beta_app(u_mean, u_std, n_components, theta_jet_std, size_samples=10000.):

    u_sample = np.random.normal(loc=u_mean, scale=u_std, size=size_samples)
    gamma_sample = np.exp(u_sample) + 1.
    large_theta_sample = get_theta_sample(gamma_sample, a=2.)
    theta_pdf = gaussian_kde(large_theta_sample)
    theta_jet = theta_pdf.resample(size=1)
    theta_components = get_samples_from_truncated_normal(theta_jet, theta_jet_std, 0., math.pi,
                                                         size=n_components)
    
    u_components = np.random.normal(loc=u_mean, scale=u_std, size=n_components)
    gamma_components = np.exp(u_components) + 1.
    
    k = np.sqrt(gamma_components ** 2. - 1.)
    return k * np.sin(theta_components) / (gamma_components - k * np.cos(theta_components))


    
    
def get_theta_sample(gamma_sample, a=2.):
    """
    Function that generates sample from theta distribution given sample from
    gamma distribution.

    :param gamma_sample:
        Sample from gamma distribution.

    :return:
        Ln of pdf of theta.
    """

    def theta_cdf(theta, beta, a=2.):
        """
        Function that returns cdf for theta given value(s) of beta.
        """
        return ((1. - beta) ** (-a) - (1. - beta * math.cos(theta)) ** (-a)) / \
               ((1. - beta) ** (-a) - 1.)

    gamma_sample = np.asarray(gamma_sample)
    # Recalculate beta distribution
    beta_sample = np.sqrt(gamma_sample ** 2. - 1.) / gamma_sample

    # Sample N uniform numbers from unif(0, 1)
    us = np.random.uniform(0, 1, size=len(beta_sample))
    # Use inverse CDF method for sampling N values of theta from theta
    # distribution
    theta_sample = np.arccos((1. - ((1. - beta_sample) ** (-a) - us *
                                    ((1. - beta_sample) ** (-a) - 1.)) **
                              (-1. / a)) / beta_sample)

    return theta_sample


def get_samples_from_truncated_normal(mean, sigma, a, b, size=10 ** 4):
    """
    Function that returns ``size`` number of samples from truncated normal
    distribution with ``mu``, ``sigma`` and truncated at [``a``, ``b``].
    """

    kwargs = {'loc': mean, 'scale': sigma}
    us = np.random.uniform(norm.cdf(a, **kwargs), norm.cdf(b, **kwargs),
                           size=size)
    return norm.ppf(us, **kwargs)