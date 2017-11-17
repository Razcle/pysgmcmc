#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import numpy as np


def leapfrog(theta, r, epsilon):
    raise NotImplementedError


def build_tree(theta, r, u, v, j, epsilon, theta_0, r_0, cost_fun):

    def base_case(theta, r, u, v, epsilon, theta_0, r_0, cost_fun, delta_max=0.01):
        theta_, r_ = leapfrog(theta, r, v * epsilon)
        n_ = u <= np.exp(cost_fun(theta_ - 0.5 * r_ * r_))
        s_ = u <= np.exp(delta_max + cost_fun(theta_) - 0.5 * r_ * r_)
        return (
            theta_, r_, theta_, r_, theta_, n_, s_,
            min(1., np.exp(cost_fun(theta_) - 0.5 * r_ * r_ - cost_fun(theta_0) + 0.5 * r_0 * r_0)),
            1
        )

    recursion_stack = []

    while j > 0:
        # recurse
        pass

    return base_case(
        theta=theta, r=r, u=u, v=v, epsilon=epsilon, theta_0=theta_0, r_0,
        cost_fun=cost_fun
    )

    # base case

