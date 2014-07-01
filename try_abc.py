#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.stats import gaussian_kde, norm


def generate_beta_app(u_mean, u_std, n_components, theta_jet_std,
                      size_samples=10000.):

    u_sample = np.random.normal(loc=u_mean, scale=u_std, size=size_samples)
    gamma_sample = np.exp(u_sample) + 1.
    large_theta_sample = get_theta_sample(gamma_sample, a=2.)
    theta_pdf = gaussian_kde(large_theta_sample)
    theta_jet = theta_pdf.resample(size=1)
    theta_components = get_samples_from_truncated_normal(theta_jet,
                                                         theta_jet_std, 0.,
                                                         math.pi,
                                                         size=n_components)

    u_components = np.random.normal(loc=u_mean, scale=u_std, size=n_components)
    gamma_components = np.exp(u_components) + 1.

    k = np.sqrt(gamma_components ** 2. - 1.)

    return k * np.sin(theta_components) / (gamma_components -
                                           k * np.cos(theta_components))




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
    return norm.ppf(us, **kwargs)[0]


def summary(gendata, data):
    """
    Returns summary statistic for given dataset and generated data.

    :param gendata:

    :param data:

    :return:
        sqrt(d(min)^2 + d(max)^2 + d(mean)^2)
    """
    return math.sqrt((np.mean(gendata) - np.mean(data)) ** 2. +
                     (np.min(gendata) - np.min(data)) ** 2. +
                     (np.max(gendata) - np.max(data)) ** 2.)


def get_summary(data, u_mean, u_std):
    gendata = generate_beta_app(u_mean, u_std, n_components=len(data),
                                theta_jet_std=0.03)
    return summary(gendata, data)


if __name__ == '__main__':

    # Load data
    data_lists = list()
    files = ['/home/ilya/work/hierarch/data/source.txt',
             '/home/ilya/work/hierarch/data/beta.txt',
             '/home/ilya/work/hierarch/data/sigma_beta.txt']
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

    source = '1510-089'
    data = betas_dict[source]

    # Gamma ~ 10 corresponds to log(G - 1) ~ 2.2
    gen_betas = generate_beta_app(2.2, 0.2, n_components=len(data),
                                  theta_jet_std=0.03, size_samples=10000)
