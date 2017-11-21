#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
import numpy as np
import random
from collections import namedtuple


def leapfrog(theta, r, epsilon):
    # XXX: Leapfrog is not interesting here and requires gradient computation
    # which is difficult to obtain for numpy
    return theta, r


def base_case(theta, r, epsilon, u, v, theta_0, r_0, cost_function, delta_max=0.01):
    theta_, r_ = leapfrog(theta, r, v * epsilon)
    n_ = int(u <= np.exp(cost_function(theta_) - 0.5 * r_ * r_))
    s_ = int(u <= np.exp(
        delta_max +
        cost_function(theta_) - 0.5 * r_ * r_ -
        cost_function(theta_0) + 0.5 * r_0 * r_0
    ))

    return (
        theta_, r_, theta_, r_, theta_, n_, s_,
        min(
            1.,
            np.exp(
                cost_function(theta_) - 0.5 * r_ * r_ -
                cost_function(theta_0) + 0.5 * r_0 * r_0
            )
        ), 1
    )


def build_tree_recursive(theta, r, u, v, j, epsilon, theta_0, r_0, cost_function):
    if j == 0:
        return base_case(theta, r, epsilon, u, v, theta_0, r_0, cost_function)
    else:
        # recursion
        (theta_left, r_left, theta_right, r_right,
         theta_, n_, s_, alpha_, nalpha_) = build_tree_recursive(
            theta, r, u, v, j - 1, epsilon, theta_0, r_0, cost_function
        )

        if s_ == 1:
            if v == -1:
                (theta_left, r_left, _, _,
                 theta__, n__, s__, alpha__, nalpha__) = build_tree_recursive(
                    theta_left, r_left, u, v, j - 1, epsilon, theta_0, r_0,
                    cost_function
                )
            else:
                (_, _, theta_right, r_right,
                 theta__, n__, s__, alpha__, nalpha__) = build_tree_recursive(
                    theta_right, r_right, u, v, j - 1, epsilon, theta_0, r_0,
                    cost_function
                )

            if random.random() <= (float(n__) / (n_ + n__)):
                theta_ = theta__

            alpha_ = alpha__ + alpha_
            nalpha_ = nalpha__ + nalpha_
            s_ = (
                s__ * int(((theta_right - theta_left) * r_left) >= 0) *
                int(((theta_right - theta_left) * r_right) >= 0)
            )

            n_ = n__ + n_
        return (
            theta_left, r_left, theta_right, r_right,
            theta_, n_, s_, alpha_, nalpha_
        )


def build_left_tree_recursive(theta, r, u, j, epsilon, theta_0, r_0, cost_fun):
    # here v is *always* - 1
    v = -1

    if j == 0:
        return base_case(
            theta, r, epsilon, u, v, theta_0, r_0, cost_fun
        )
    else:
        (theta_left, r_left, theta_right, r_right,
         theta_, n_, s_, alpha_,
         nalpha_) = build_left_tree_recursive(
            theta, r, u, j - 1, epsilon, theta_0, r_0, cost_fun
        )

        if s_ == 1:
            (theta_left, r_left, _, _, theta__,
             n__, s__, alpha__,
             nalpha__) = build_left_tree_recursive(
                theta_left, r_left, u, j - 1, epsilon, theta_0, r_0, cost_fun
            )

            if random.random() <= (float(n__) / (n_ + n__)):
                theta_ = theta__

            alpha_ = alpha__ + alpha_
            nalpha_ = nalpha__ + nalpha_
            s_ = (
                s__ * int(((theta_right - theta_left) * r_left) >= 0) *
                int(((theta_right - theta_left) * r_right) >= 0)
            )

            n_ = n__ + n_
        return (
            theta_left, r_left, theta_right, r_right,
            theta_, n_, s_, alpha_, nalpha_
        )


def build_right_tree_recursive(theta, r, u, j, epsilon, theta_0, r_0, cost_fun):
    # here v is *always* 1
    v = 1

    if j == 0:
        return base_case(
            theta, r, epsilon, u, v, theta_0, r_0, cost_fun
        )

    else:
        (theta_left, r_left, theta_right, r_right,
         theta_, n_, s_, alpha_,
         nalpha_) = build_right_tree_recursive(
            theta, r, u, j - 1, epsilon, theta_0, r_0, cost_fun
        )

        if s_ == 1:
            (_, _, theta_right, r_right, theta__,
             n__, s__, alpha__,
             nalpha__) = build_right_tree_recursive(
                theta_right, r_right, u, j - 1, epsilon, theta_0, r_0, cost_fun
            )

            if random.random() <= (float(n__) / (n_ + n__)):
                theta_ = theta__

            alpha_ = alpha__ + alpha_
            nalpha_ = nalpha__ + nalpha_
            s_ = (
                s__ * int(((theta_right - theta_left) * r_left) >= 0) *
                int(((theta_right - theta_left) * r_right) >= 0)
            )

            n_ = n__ + n_
        return (
            theta_left, r_left, theta_right, r_right,
            theta_, n_, s_, alpha_, nalpha_
        )

# XXX: Return address?
Stackframe = namedtuple("Stackframe", ["call_arguments", "local_variables", "return_address"])


class CallArguments(tuple):
    def __new__(cls, cost_fun, theta=None, r=None, u=None, j=None,
                epsilon=None, theta_0=None, r_0=None):
        return super().__new__(cls, tuple(theta, r, u, j, epsilon, theta_0, r_0, cost_fun))

# XXX: Type for local variables?


def build_right_tree_iterative(theta, r, u, j, epsilon, theta_0, r_0, cost_fun):
    v = 1
    stack = [
        Stackframe(CallArguments(theta=theta, r=r, u=u, j=j, epsilon=epsilon,
                                 theta_0=theta_0, r_0=r_0, cost_fun=cost_fun),
                   local_variables=(), return_address=None) # XXX Address?
    ]

    while stack:
        frame = stack.pop()



def build_tree_recursive_combined(theta, r, u, v, j, epsilon, theta_0, r_0, cost_fun):
    if v == 1:
        return build_right_tree_recursive(theta, r, u, j, epsilon, theta_0, r_0, cost_fun)
    else:
        return build_left_tree_recursive(theta, r, u, j, epsilon, theta_0, r_0, cost_fun)


def test_random():
    from pysgmcmc.diagnostics.objective_functions import (
        gmm1_log_likelihood, to_negative_log_likelihood
    )
    u = random.random()
    v = random.choice((-1, 1))
    theta = random.random() * random.randint(-3000, 3000)
    r = random.random() * random.randint(-2000, 2000)
    theta_0 = random.random() * random.randint(-3000, 3000)
    r_0 = random.random() * random.randint(-2000, 2000)
    epsilon = random.random()
    cost_function = to_negative_log_likelihood(gmm1_log_likelihood)

    j = np.random.randint(0, 10)
    print(j)

    n_tests = 5
    while n_tests > 0:
        results_recursive = build_tree_recursive(
            theta, r, u, v, j, epsilon, theta_0, r_0, cost_function
        )

        results_split = build_tree_recursive_combined(
            theta, r, u, v, j, epsilon, theta_0, r_0, cost_function
        )

        assert results_recursive == results_split
        n_tests -= 1
