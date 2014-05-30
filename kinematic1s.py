#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import special
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.special import gamma
import sys
sys.path.append('/home/ilya/work/upstream/emcee')
import emcee



class LnPost(object):
    """
    Class that represents log of posterior density for kinematic model.
    """
    def __init__(self, beta_obs, sigma_beta_obs, alpha_max=None, beta_max=None,
                 c_min=None, c_max=None, d_min=None, d_max=None, s_t=0.01,
                 r_t=0.01):
        beta_obs_min = min(beta_obs)
        sigma_beta_obs_min = sigma_beta_obs[beta_obs.index(beta_obs_min)]
        beta_obs_max = max(beta_obs)
        sigma_beta_obs_max = sigma_beta_obs[beta_obs.index(beta_obs_max)]
        if alpha_max is None:
            alpha_max = 10.
        if beta_max is None:
            beta_max = 10.
        if c_min is None:
            c_min = 1.
        if d_min is None:
            d_min = beta_obs_max + 3. * sigma_beta_obs_max
        if c_max is None:
            c_max = d_min
        if d_max is None:
            d_max = 100.
        self._lnlike = LnLike(beta_obs, sigma_beta_obs)
        self._lnpr = LnPrior(alpha_max=alpha_max, beta_max=beta_max,
                             c_min=c_min, c_max=c_max, d_min=d_min, d_max=d_max,
                             s_t=s_t, r_t=r_t)

    def __call__(self, p):
        lnpr = self._lnpr(p)
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

    def __init__(self, beta_obs, sigma_beta_obs):
        self.beta_obs = beta_obs
        self.sigma_beta_obs = sigma_beta_obs
        self.n_c = len(beta_obs)
        assert(len(beta_obs) == len(sigma_beta_obs))

    def __call__(self, p):
        """
        Returns ln of likelihood for kinematic model.

        :param p:
        Parameters of model. [\alpa, \beta, c, d, \mu_t, \tau_t, \G_{i}, \t_{i}]

        :return:
            Ln of likelihood function.
        """

        p = np.asarray(p)
        # k1: \mu_t ~ P_{KDE}(params of genbeta pdf: \alpha, \beta, c, d)
        # Use abs() as negative values raise ValueError but have zero prior
        # prob.
        gamma_sample = get_samples_from_genbeta(abs(p[0]), abs(p[1]), p[2], p[3],
                                                size=10 ** 4)
        large_theta_sample = get_theta_sample(gamma_sample, a=2.)
        try:
            kde = gaussian_kde(large_theta_sample)
            k1 = math.log(kde(p[4]))
        except ValueError:
            print "ValueError in LnLike.__call__ in kde of large_theta_sample"
            print p
            k1 = float("-inf")


        # k2: \G_{i} ~ genBeta(\alpha, \beta, c, d)
        try:
            k2 = np.sum(vec_lngenbeta(p[6: 6 + self.n_c], p[0], p[1], p[2], p[3]))
        except ValueError:
            print "ValueError in LnLike.__call__ in vec_lngenbeta()"
            k2 = float("-inf")
            print p

        # k3: \t_{i} ~ trNorm(\mu_t, \tau_t)
        try:
            k3 = np.sum(vec_ln_normal_trunc(p[6 + self.n_c: 6 + 2 * self.n_c],
                                            p[4], p[5], 0., math.pi))
        except ValueError:
            print "ValueError in LnLike.__call__ in vec_ln_normal_trunc()"
            k3 = float("-inf")
            print p

        # k4: beta^{obs}_{i} ~ Norm(beta_obs_{i}, model(\G_{i}, \t_{i}),
        # \tau_obs_i))
        k4 = np.sum(self.ln_of_normpdf(self.beta_obs,
                                       loc=model(p[6: 6 + self.n_c],
                                       p[6 + self.n_c: 6 + 2 * self.n_c]),
                                       scale=self.sigma_beta_obs))

        return k1 + k2 + k3 + k4


class LnPrior(object):
    def __init__(self, alpha_max=None, beta_max=None, c_min=None, c_max=None,
                 d_min=None, d_max=None, s_t=None, r_t=None):
        self.alpha_max = alpha_max
        self.beta_max = beta_max
        self.c_min = c_min
        self.c_max = c_max
        self.d_min = d_min
        self.d_max = d_max
        self.s_t = s_t
        self.r_t = r_t

    def __call__(self, p):
        """
        Prior on parameters.

        :param p:
            Parameters of model. [\alpa, \beta, c, d, \mu_t, \tau_t, \G_{i},
            \t_{i}]
        :return:
            Log of prior density for given parameter ``p``.
        """

        p = np.asarray(p)
        # \tau_t ~ Gamma(s_t, r_t)
        result = vec_ln_gamma(p[5], self.s_t, self.r_t)

        # \alpha ~ Unif(0, \alpha_max)
        if (self.alpha_max < p[0]) or (p[0] < 0):
            result = float("-inf")

        # \beta ~ Unif(0, \beta_max)
        if (self.beta_max < p[1]) or (p[1] < 0):
            result = float("-inf")

        # c ~ Unif(1, \c_max)
        if (self.c_max < p[2]) or (p[2] < 1):
            result = float("-inf")

        # d ~ Unif(d_min, d_max)
        if (self.d_max < p[3]) or (p[3] < self.d_min):
            result = float("-inf")

        return result


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


def get_samples_from_genbeta(alpha, beta, c, d, *args, **kwargs):
    return (d - c) * np.random.beta(alpha, beta, *args, **kwargs) + c


def get_samples_from_normal(mean, tau, size=10 ** 4):
    return np.random.normal(loc=mean, scale=1./math.sqrt(tau), size=size)


def get_samples_from_shifted_lognormal(mean, sigma, shift, size=10 ** 4):
    """
    Function that returns ``size`` number of samples from lognormal distribution
    with ``mu``, ``sigma`` and shifted by ``shift``.
    """
    return np.random.lognormal(mean=mean, sigma=sigma, size=size) + float(shift)


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


def get_pdf_of_theta_given_gamma(mean, tau, size=10 ** 4, a=2.):
    """
    Function that returns callable pdf of theta distribution given parameters
    of gamma distribution (that is shifted lognormal).
    """
    gamma_sample = get_samples_from_normal(mean, tau, size=size)
    theta_sample = get_theta_sample(gamma_sample, a=a)
    return gaussian_kde(theta_sample)


def vec_lnlognorm(x, mu, sigma, shift=0.):
    """
    Vectorized (natural logarithm of) shifted lognormal distribution.
    Used for modelling \Gamma_{j} with mean = ``shift``, ``mu`` = 0 and
    sigma ~ 0.1/1.
    """

    x_ = np.where(0 < (x - shift), (x - shift), 1)
    result1 = -np.log(np.sqrt(2. * math.pi) * x_ * sigma) - (np.log(x_) - mu) \
                                                            ** 2 / (2. * sigma ** 2)
    result = np.where(0 < (x - shift), result1, float("-inf"))

    return result


def vec_lngenbeta(x, alpha, beta, c, d):
    """
    Vectorized (natural logarithm of) Beta distribution with support (c,d).
    A.k.a. generalized Beta distribution.
    """

    x_ = np.where((c < x) & (x < d), x, 1)

    result1 = -math.log(special.beta(alpha, beta)) - (alpha + beta - 1.) * \
                                                     math.log(d - c) + (alpha - 1.) * np.log(x_ - c) + (beta - 1.) * \
              np.log(d - x_)
    result = np.where((c < x) & (x < d), result1, float("-inf"))

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
    beta_obs = betas_dict[source]
    sigma_beta_obs = beta_sigmas_dict[source]
    tau_obs = [1./sigma**2. for sigma in beta_sigmas_dict[source]]

    lnpost = LnPost(beta_obs, sigma_beta_obs)
    ndim = 6 + 2 * len(beta_obs)
    nwalkers = 250
    # Parameters of model. [\alpa, \beta, c, d, \mu_t, \tau_t, \G_{i}, \t_{i}]
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


