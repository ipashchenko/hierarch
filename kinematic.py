#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import special
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.special import gamma


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
    def ln_of_normpdf2(x, loc=0., logscale=0.):
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

    @staticmethod
    def ln_of_normpdf1(x, loc=0., scale=0.):
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
        return -0.5 * math.log(2. * math.pi) - math.log(scale) - \
               (x - loc) ** 2 / (2. * scale ** 2.)

    @staticmethod
    def ln_of_normpdf3(x, loc=0., tau=0.):
        """
        Returns ln of pdf for normal distribution at points x.

        :param x:
            Value (or array of values) for which to calculate the result.

        :param loc (optional):
            Mean of the distribution.

        :param tau (optional):
            Precision of the distribution.

        :return:
            Log of pdf for normal distribution with specified parameters. Value
            or array of values (depends on ``x``).
        """
        return -0.5 * math.log(2. * math.pi) + 0.5 * math.log(tau) - \
               tau * (x - loc) ** 2 / 2.

    def __init__(self, beta_ij):
        self._beta_ij = beta_ij
        self.n_s = len(beta_ij)

    def __call__(self, p):
        """
        Returns ln of likelihood for kinematic model.

        :param p:
        Parameters of model. [alpha, beta, \gamma_{max}, \mu_g_{j}, \mu_t_{j},
        \tau_g_{j}, \tau_t_{j}, s_g, r_g, s_t, r_t, j=1,...,N_{source}]

        :return:
            Ln of likelihood function.
        """

        n_s = self.n_s
        # Sums for j=1 to #sources

        # k1: \mu_g_{j} ~ genBeta(alpha, beta, 2, \gamma_{max})
        k1 = np.sum(vec_lngenbeta(p[3: 3 + n_s], p[0], p[1], 2, p[2]))

        # k2: \tau_g_{j} ~ Gamma(s_g, r_g)
        k2 = np.sum(vec_ln_gamma(p[6 + 2 * n_s: 6 + 3 * n_s], p[3], p[4]))

        # k3: \tau_t_{j} ~ Gamma(s_t, r_t)
        k3 = np.sum(vec_ln_gamma(p[6 + 3 * n_s: 6 + 4 * n_s], p[5], p[6]))

        # k4: \mu_t_{j} ~ P_{KDE}(params of shifted lognormal pdf: mean=0,
        #                         \tau_t_{j}, shift=\mu_g_{j})
        kde_lnprob_list = list()
        for j in range(n_s):
            kde = get_pdf_of_theta_given_gamma(0, p[6 + 2 * n_s + j], p[6 + j],
                                               size=10. ** 4, a=2.)
            kde_lnprob_list.append(math.log(kde(p[6 + n_s + j])))
        k4 = sum(kde_lnprob_list)

        # k5: gamma_{ij} ~ logNorm(mean=0, tau=\tau_g_{j}, shift=\mu_g_{j})
        k5 = np.sum(vec_lnlognorm())

        # k6: theta_{ij} ~ Norm(mean=\mu_t_{j}, tau=\tau_t_{j}, 0, \pi)

        # k7: beta^{obs}_{ij} ~ Norm(mean=model(gamma_{ij}, theta_{ij}),
        #                            tau=tau^{obs}_{ij}))
        k7 = sum([np.sum(self.ln_of_normpdf3(self.beta_obs_ij[j],
                                             loc=model(gamma_ij, theta_ij),
                                             tau=tau_obs_ij)) for j, theta in
                 enumerate(p[3:])])




        # Double sums for j=1 to #sources & for i=1 to #obs for current source
        kk1 = sum([np.sum(vec_lnlognorm())])

        #k = sum([np.sum(self.ln_of_normpdf2(self.y_ij[j], loc=theta,
        #                                   logscale=p[2])) for j, theta in
        #         enumerate(p[3:])])
        #return np.sum(self.ln_of_normpdf2(p[3:], loc=p[0], logscale=p[1])) + k


def lnpr(p):
    """
    Prior on parameters.

    :param p:
        Parameters of model. [alpha, beta, \Gamma_{max}, \Gamma_{j}, \Theta_{j},
          s_Gj, r_Gj, s_Tj, r_Tj, j=1,...,N_{source}]

    :return:
        Log of prior density for given parameter ``p``.
    """

    result = 0
    if p[0] < 0 or p[1] < 0 or p[2] < 2:
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
    k = math.sqrt(gamma ** 2. - 1.)
    return k * math.sin(theta) / (gamma - k * math.cos(theta))


def get_samples_from_shifted_lognormal(mean, sigma, shift, size=10. ** 4):
    """
    Function that returns ``size`` number of samples from lognormal distribution
    with ``mu``, ``sigma`` and shifted by ``shift``.
    """
    return np.random.lognormal(mean=mean, sigma=sigma, size=size) + float(shift)


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

# TODO: Calculate theta_pdf for whole distribution of gamma/beta and fit with
# KDE, then, for each object, get theta_pdf_i using importance sampling with
# KDE as importance function


def get_pdf_of_theta_given_gamma(mean, sigma, shift, size=10. ** 4, a=2.):
    """
    Function that returns callable pdf of theta distribution given parameters
    of gamma distribution (that is shifted lognormal).
    """
    gamma_sample = get_samples_from_shifted_lognormal(mean=mean, sigma=sigma,
                                                shift=shift, size=size)
    theta_sample = get_theta_sample(gamma_sample, a=a)
    return gaussian_kde(theta_sample)


def vec_lnlognorm(x, mu, sigma, shift=0.):
    """
    Vectorized (natural logarithm of) shifted lognormal distribution.
    Used for modelling \Gamma_{j} with mean = ``shift``, ``mu`` = 0 and
    sigma ~ 0.1/1.
    """

    x_ = np.where(0 < (x - shift), (x - shift), 1)
    result1 = -np.log(np.sqrt(2. * math.pi) * x_ * sigma) - (np.log(x_) - mu)\
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
    result1 = s * math.log(r) - math.log(gamma(s)) + (s - 1.) * np.log(x_) -\
              r * x_
    result = np.where(0 < x, result1, float("-inf"))

    return result


def ln_normal_trunc(x, mean, sigma, a, b):
    """
    Function that returns log of truncated at [a, b] normal distribution with
    mean ``mean`` and variance ``sigma**2``.
    """
    x_ = np.where((a < x) & (x < b), x, 1)
    k = math.log(norm.cdf((b - mean) / sigma) -
                 norm.cdf((a - mean) / sigma))
    result1 = -math.log(sigma) - 0.5 * math.log(2. * math.pi) -\
           0.5 * ((x_ - mean) / sigma) ** 2. - k
    result = np.where((a < x) & (x < b), result1, float("-inf"))

    return result


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
