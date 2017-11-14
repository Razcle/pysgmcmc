import tensorflow as tf

from pysgmcmc.samplers.base_classes import MCMCSampler
from pysgmcmc.tensor_utils import (
    vectorize, unvectorize, safe_divide, safe_sqrt,
)
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule
from pysgmcmc.tensor_utils import choice


class NUTSSampler(MCMCSampler):
    def __init__(self, params, cost_fun, batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.01), seed=None,
                 session=tf.get_default_session(), dtype=tf.float64):
        s = 1

        # XXX Initialize
        theta_minus = r_minus = epsilon = theta = j = u = r_0 = None
        theta_plus = r_plus = None

        while s == 1:
            v_j = choice((-1, 1))

            if v_j == -1:
                (theta_minus, r_minus, _, _,
                 theta_prime, n_prime, s_prime,
                 alpha, n_alpha) = self.build_tree(
                    theta_minus, r_minus, u, v_j, j, epsilon * theta, r_0
                )
            else:
                (_, _, theta_plus, r_plus,
                 theta_prime, n_prime, s_prime,
                 alpha, n_alpha) = self.build_tree(
                    theta_plus, r_plus, u, v_j, j, epsilon * theta, r_0
                )
             with probability(...):
                 theta <- theta_prime

    def _costs_for(self, param_values):
        assignments = []
        # XXX Cache and re-assign old parameter values?
        for param, param_value in zip(self.params, param_values):
            assignments.append(tf.assign(param, param_value))
        self.session.run(assignments)

        return self.session.run(self.Cost)

    def leapfrog(self, theta, r, epsilon):
        raise NotImplementedError

    def build_tree(self, theta, r, u, v, j, epsilon, theta_0, r_0):
        # XXX: Transform conditions to using tf.cond..

        costs_theta_0 = self._costs_for(theta_0)
        if j == 0:
            theta_prime, r_prime = self.leapfrog(theta, r, v * epsilon)

            costs_theta_prime = self._costs_for(theta_prime)

            n_prime = tf.cast(
                tf.less_equal(
                    u,
                    tf.exp(costs_theta_prime - 0.5 * r_prime * r_prime)
                ), tf.int32
            )

            s_prime = tf.cast(
                tf.less_equal(
                    u,
                    self.delta_max + costs_theta_prime - 0.5 * r_prime * r_prime
                ), tf.int32
            )

            return (
                theta_prime, r_prime, theta_prime, r_prime, theta_prime,
                n_prime, s_prime,
                tf.minimum(
                    1,
                    tf.exp(costs_theta_prime - 0.5 * r_prime * r_prime - costs_theta_0 + 0.5 * r_0 * r_0)
                ),
                1
            )
        else:
            tree_results = self.build_tree(
                theta, r, u, v, j - 1, epsilon, theta_0, r_0
            )

            (theta_minus, r_minus, theta_plus,
             r_plus, theta_prime, n_prime,
             s_prime, alpha_prime, n_alpha_prime) = tree_results

            if s_prime == 1:
                if v == -1:
                    (theta_minus, r_minus, _, _,
                     theta_prime_prime, n_prime_prime,
                     s_prime_prime, alpha_prime_prime,
                     n_alpha_prime_prime) = self.build_tree(
                        theta_minus, r_minus, u, v, j - 1,
                        epsilon, theta_0, r_0
                    )
                else:
                    (_, _, theta_plus, r_plus,
                     theta_prime_prime, n_prime_prime,
                     s_prime_prime, alpha_prime_prime,
                     n_alpha_prime_prime) = self.build_tree(
                        theta_plus, r_plus, u, v, j - 1,
                        epsilon, theta_0, r_0
                    )

                # XXX: With probability n'' / (n' + n'')

                alpha_prime = alpha_prime + alpha_prime_prime
                n_alpha_prime = n_alpha_prime + n_alpha_prime_prime
                n_prime = n_prime + n_prime_prime

                return (
                    theta_prime, r_prime,
                    theta_plus, r_plus,
                    theta_prime, n_prime,
                    s_prime, alpha_prime, n_alpha_prime
                )
