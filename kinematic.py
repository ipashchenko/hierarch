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
    np.hist(get_theta_sample(dgamma, mean=2, sigma=0.5, size=10000)*180./math.pi, bins=50, normed=True)
