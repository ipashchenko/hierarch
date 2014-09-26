#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import special
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.special import gamma
import sys
sys.path.append('/home/ilya/code/upstream/emcee')
import emcee



class LnPost(object):
    """
    Class that represents log of posterior density for kinematic model.
    """
    def __init__(self, beta_obs, sigma_beta_obs, prags=[], prkwargs={},
                likargs=[], likkwargs={}):
        #beta_obs_min = min(beta_obs)
        #sigma_beta_obs_min = sigma_beta_obs[beta_obs.index(beta_obs_min)]
        #beta_obs_max = max(beta_obs)
        #sigma_beta_obs_max = sigma_beta_obs[beta_obs.index(beta_obs_max)]
        self._lnlike = LnLike(beta_obs, sigma_beta_obs, *likargs, **likkwargs)
        self._lnpr = LnPrior(*prags, **prkwargs)
        self._lnpr.n_c = self._lnlike.n_c

    def __call__(self, p):
        lnpr = self._lnpr(p)
        # If prior is ``-inf`` then don't calculate likelihood
        if lnpr == float("-inf"):
            result = float("-inf")
            print "zero prob. prior for parameter :"
            print p
        else:
            result = lnpr + self._lnlike(p)
        return result


class LnLike(object):
    """
    Class that represents ln of likelihood for kinematic model.
    """

    @staticmethod
    def ln_of_normpdf(x, loc=0., scale=0.):
        """
        Returns ln of pdf for normal distribution at points x.

        :param x:
            Value (or array of values) for which to calculate the result.

        :param loc (optional):
            Mean of the distribution.

        :param scale (optional):
            Std of the distribution.

        :return:
            Log of pdf for normal distribution with specified parameters. Value
            or array of values (depends on ``x``).
        """
        scale = np.asarray(scale)
        loc = np.asarray(loc)
        x = np.asarray(x)
        return -0.5 * math.log(2. * math.pi) - np.log(scale) - \
               (x - loc) ** 2. / (2. * scale ** 2.)

    def __init__(self, beta_obs, sigma_beta_obs, gamma_max=None):
        self.beta_obs = beta_obs
        self.sigma_beta_obs = sigma_beta_obs
        self.gamma_max=gamma_max
        self.n_c = len(beta_obs)
        assert(len(beta_obs) == len(sigma_beta_obs))

    def __call__(self, p):
        """
        Returns ln of likelihood for kinematic model.

        :param p:
        Parameters of model. [\Gamma_0, \tau_g, \theta_0, \tau_t, \Gamma_{i},
        \theta_{i}].

        :return:
            Ln of likelihood function.
        """

        p = np.asarray(p)
        # likelihood: beta^{obs}_{i} ~ Norm(model(\G_{i}, \t_{i}), \tau_obs_i))
        r1 = np.sum(self.ln_of_normpdf(self.beta_obs,
                                       loc=model(p[4: 4 + self.n_c],
                                       p[4 + self.n_c: 4 + 2 * self.n_c]),
                                       scale=self.sigma_beta_obs))
        return r1


# TODO: add prior on absence of reverse moving: delta_t < theta_0
class LnPrior(object):
    def __init__(self, gamma_max=None, s_g=None, r_g=None, s_t=None, r_t=None):
        self.gamma_max = gamma_max
        self.s_g = s_g
        self.r_g = r_g
        self.s_t = s_t
        self.r_t = r_t

    def __call__(self, p):
        """
        Prior on parameters.

        :param p:
            Parameters of model. [\Gamma_0, \tau_g, \theta_0, \tau_t,
            \Gamma_{i},\theta_{i}].

        :return:
            Log of prior density for given parameter ``p``.
        """

        p = np.asarray(p)

        # \Gamma_0 ~ unif(1., gamma_max)
        r1 = vec_ln_unif(p[0], 1., self.gamma_max)

        # \tau_g ~ Gamma(s_g, r_g)
        r2 = vec_ln_gamma(p[1], self.s_g, self.r_g)

        # \tau_t ~ Gamma(s_t, r_t)
        r3 = vec_ln_gamma(p[3], self.s_t, self.r_t)

        # \theta_0 ~ KDE(\Gamma_0, \tau_g)
        gamma_sample = get_samples_from_truncated_normal(p[0], 1. / p[1] ** 2.,
                                                         1., self.gamma_max)
        large_theta_sample = get_theta_sample(gamma_sample, a=2.)
        try:
            kde = gaussian_kde(large_theta_sample)
            r4 = math.log(kde(p[2]))
        except ValueError:
            print "ValueError in LnLike.__call__ in kde of large_theta_sample"
            print p
            r4 = float("-inf")

        # \Gamma_{i} ~ trNorm(\Gamma_0, \tau_g)
        r5 = np.sum(vec_ln_normal_trunc(p[4: 4 + self.n_c], p[0],
                                        1. / p[1] ** 2., 1., self.gamma_max))

        delta = 0.5 / math.sqrt(p[3])
        # \theta_{i} ~ Unif(\theta_0 - delta, \theta_0 + delta)
        r6 = np.sum(vec_ln_unif(p[4 + self.n_c: 4 + 2 * self.n_c], p[2] - delta,
                                p[2] + delta))

        return r1 + r2 + r3 + r4 + r5 + r6


def model(gamma, theta):
    """
    Model used.

    :param gamma:
        Lorenz factor.

    :param theta:
        Angle to LOS.

    :return:
        Apparent speed (in units of speed of light)

    \beta_{app} = sqrt(\gamma ** 2 - 1) * sin(\theta) /
                    (\gamma - sqrt(\gamma ** 2 - 1) * cos(\theta))
    """
    k = np.sqrt(gamma ** 2. - 1.)
    return k * np.sin(theta) / (gamma - k * np.cos(theta))


def get_samples_from_truncated_normal(mean, sigma, a, b, size=10 ** 4):
    """
    Function that returns ``size`` number of samples from truncated normal
    distribution with ``mu``, ``sigma`` and truncated at [``a``, ``b``].
    """

    kwargs = {'loc': mean, 'scale': sigma}
    us = np.random.uniform(norm.cdf(a, **kwargs), norm.cdf(b, **kwargs),
                           size=size)
    return norm.ppf(us, **kwargs)


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


def vec_ln_unif(x, a, b):
    """
Vectorized (natural logarithm of) uniform distribution on [a, b].
    """

    result1 = -np.log(b - a)
    result = np.where((a <= x) & (x <= b), result1, float('-inf'))

    return result


def vec_ln_gamma(x, s, r):
    """
    Vectorized (natural logarithm of) Gamma distribution with shape and rate
    parameters ``s`` & ``r``.
    """
    assert((s > 0) & (r > 0))
    x_ = np.where(0 < x, x, 1)
    result1 = s * math.log(r) - math.log(gamma(s)) + (s - 1.) * np.log(x_) - \
              r * x_
    result = np.where(0 < x, result1, float("-inf"))

    return result


def vec_ln_normal_trunc(x, mean, sigma, a, b):
    """
    Function that returns log of truncated at [a, b] normal distribution with
    mean ``mean`` and variance ``sigma**2``.
    """
    x_ = np.where((a < x) & (x < b), x, 1)
    k = math.log(norm.cdf((b - mean) / sigma) -
                 norm.cdf((a - mean) / sigma))
    result1 = -math.log(sigma) - 0.5 * math.log(2. * math.pi) - \
              0.5 * ((x_ - mean) / sigma) ** 2. - k
    result = np.where((a < x) & (x < b), result1, float("-inf"))

    return result


def generate_data(gamma_0, tau_g, theta_0=None, delta_t=1.3 * math.pi / 180.,
                  n_c=None, gamma_max=50.):
    """
    Generate superluminal velocities for ``n_c`` components.

    :param gamma_0:
    :param tau_g:
    :param theta_0:
    :param s_t:
    :param n_c:

    :return:
        List of ``n_c`` values of \beta
    """

    if theta_0 is None:
        # Prepare sample of \Gamma for simulating sample of \theta
        gamma_sample = get_samples_from_truncated_normal(gamma_0,
                                                         math.sqrt(1. / tau_g),
                                                         1., gamma_max)
        theta_sample = get_theta_sample(gamma_sample)
        theta_pdf = gaussian_kde(theta_sample)
        # Choose mean LOS angle
        theta_0 = theta_pdf.resample(size=1)
    # Simulate individual components angles
    theta_comps = abs(np.random.uniform(theta_0 - delta_t, theta_0 + delta_t,
                                    size=n_c))
    # Simulate individual components \gamma
    gamma_comps = get_samples_from_truncated_normal(gamma_0,
                                                    math.sqrt(1. / tau_g), 1.,
                                                    gamma_max, size=n_c)
    return 180. * theta_0 / math.pi, model(gamma_comps, theta_comps)


def load_data(source):

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

    beta_obs = betas_dict[source]
    sigma_beta_obs = beta_sigmas_dict[source]
    tau_obs = [1./sigma**2. for sigma in beta_sigmas_dict[source]]

    return beta_obs, sigma_beta_obs


if __name__ == '__main__':


    beta_obs, sigma_beta_obs = load_data('some source')

    gamma_max = {'gamma_max': 50.}
    lnpost = LnPost(beta_obs, sigma_beta_obs, likkwargs=gamma_max,
                    prkwargs=gamma_max)
    ndim = 6 + 2 * len(beta_obs)
    nwalkers = 250
    # Parameters of model. [\Gamma_0, \tau_g, \theta_0, \tau_t,
    # \Gamma_{i},\theta_{i}].
    p0 = emcee.utils.sample_ball([5., 5., 1.2, 50., 0.1, 0.02, 10., 10.,
                                  10., 10., 10., 10., 10., 10., 10., 10., 0.1,
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                 [1., 1., 0.02, 5., 0.01, 0.001, 2., 2., 2., 2.,
                                  2., 2., 2., 2., 2., 2., 0.001, 0.001, 0.001,
                                  0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                                  0.001], nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost)
    pos, prob, state = sampler.run_mcmc(p0, 500)
    sampler.reset()
    sampler.run_mcmc(pos, 1000, rstate0=state)


