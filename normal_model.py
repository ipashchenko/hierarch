#!/usr/bin python
# -*- coding: utf-8 -*-

import numpy as np


class LnPost(object):
    """
    Class that represents ln of posterior density for normal model.
    """
    def __init__(self, yij):
        self._lnlike = LnLike(yij)

    def __call__(self, *args, **kwargs):
        pass


class LnLike(object):
    """
    Class that represents ln of likelihood for normal model.
    """
    def __init__(self, yij):
        pass


