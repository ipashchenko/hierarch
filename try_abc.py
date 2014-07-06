#!/usr/bin python2
# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/ilya/work/emcee')
import emcee
import numpy as np
import math
from scipy.stats import gaussian_kde, norm


def abc_lnprob(p, data, eps, summary_diff_fn, gamma_max=100.):
    """
    Function that returns ln of prior probability for given parameters ``p`` if
    summary statistic for generated data lies within tolerance ``eps`` from
    summary statistic of real data ``data`` or ``-inf`` if it is not the case.

    :param p:
        Parameter vector. [\gamma_mean, \gamma_std, \theta, \delta_theta], where
        \gamma_mean & \gamma_std - mean & std of truncated at [1, ``gamma_max``]
        normal distribution, \theta - mean angle of jet to LOS and \delta_theta
        - range of angles for components around some \theta.

    :param data:
        Data of observed superluminal speeds.

    :param eps:
        Tolerance.

    :param summary_diff_fn:
        Function that given generated and real data returns difference of some
        summary statistic for this data sets.

    :return:
        ln of prior density for ``p`` if output of summary_diff_fn is less then
        tolerance ``eps`` and ``-inf`` otherwise.
    """

    # Number of jet components
    n_comps = len(data)
    # Generate big sample of \gamma for constructing pdf for \theta_jet
    gamma_sample = get_samples_from_truncated_normal(p[0], p[1], 1, gamma_max)
    prior_theta_sample = get_theta_sample(gamma_sample, a=2.)

    try:
        theta_prior = gaussian_kde(prior_theta_sample)
    except ValueError:
        print "Using mean theta for jet"
        theta_jet = np.mean(prior_theta_sample)

    theta_comps = np.random.uniform(low=p[2] - 0.5 * p[3],
                                    high=p[2] + 0.5 * p[3],
                                    size=n_comps)

    # Generating data
    gamma_comps = get_samples_from_truncated_normal(p[0], p[1], 1, gamma_max,
                                                    size=n_comps)
    k = np.sqrt(gamma_comps ** 2. - 1.)
    gen_data = k * np.sin(theta_comps) / (gamma_comps - k * np.cos(theta_comps))

    if p[1] <= 0:
        result = float("-inf")
        print "Non-positive scale"
    else:
        summary_ = summary(gen_data, data)
        if summary_ < eps:
            # Calculating ln of prior for ``p``
            result = lnpr(p)
        else:
            result = float("-inf")

    return result


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
                     (np.max(gendata) - np.max(data)) ** 2.) / (math.sqrt(3.) *
                                                                len(data))


def load_data(name):

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

    data = betas_dict[name]
    return data


if __name__ == '__main__':

    data = load_data()
   # # Gamma ~ 10 corresponds to log(G - 1) ~ 2.2
   # gen_betas = generate_beta_app(2.2, 0.2, n_components=len(data),
   #                               theta_jet_std=0.03, size_samples=10000)
   # eps = 1.2
   # ndim = 2
   # nwalkers = 250
   # p0 = emcee.utils.sample_ball([2.1, 0.08], [0.4, 0.02], nwalkers)
   # sampler = emcee.EnsembleSampler(nwalkers, ndim, indicator, args=[data, eps])
   # pos, prob, state = sampler.run_mcmc(p0, 500)
   # sampler.reset()
   # sampler.run_mcmc(pos, 1000, rstate0=state)

    # Using MH
    ndim = 2
    eps = 1.5
    p0 = [2.5, 0.80]
    cov = [[0.1, 0], [0, 0.01]]
    mhsampler = emcee.mh.MHSampler(cov, ndim, indicator, args=[data, eps])
    pos, prob, state = mhsampler.run_mcmc(p0, 10000)
    mhsampler.reset()
    mhsampler.run_mcmc(pos, 100000, rstate0=state)
