import tensorflow as tf

from pysgmcmc.samplers.base_classes import MCMCSampler
from pysgmcmc.tensor_utils import (
    indicator, choice
)
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule

# XXX: Missing: epsilon estimation; mass adaption; supporting non-trivial shapes

# XXX: We have two limitations:
# - 1-d shapes are assumed for params in a number of places
# - we need the cost function to map params to a tf.Tensor,
# so this will not work with our BNN for now


class NUTSSampler(MCMCSampler):
    def __init__(self, params, cost_fun,
                 stepsize_schedule=ConstantStepsizeSchedule(0.01), seed=None,
                 session=tf.get_default_session(), dtype=tf.float64):
        self.params = params
        self.cost_fun = cost_fun

        self.Cost = cost_fun(self.params)
        self.session = session

        self.delta_max = tf.constant(1.0, dtype=dtype)

        # XXX: Support less plain shapes as well?
        r_0 = tf.random_normal(shape=(), dtype=dtype)

        u = tf.random_uniform(
            shape=(),
            minval=0.,
            maxval=tf.exp(
                self.cost_fun([param - 0.5 * r_0 * r_0 for param in params])
            )
        )

        epsilon = tf.constant(0.01, dtype=dtype, name="epsilon")

        # Used during while loop
        s = tf.constant(1, dtype=tf.int32, name="s")
        n = tf.constant(1, dtype=tf.int32, name="n")
        j = tf.constant(0, dtype=tf.int32, name="j")

        def loop_body(s, n, j, theta_last, theta_minus, r_minus,
                      theta_plus, r_plus, r_0, u):
            print("BODY")
            v_j = choice((-1, 1))

            (theta_minus_tree, r_minus_tree,
             theta_plus_tree, r_plus_tree,
             theta_prime, n_prime, s_prime, alpha, n_alpha) = tf.cond(
                pred=tf.equal(v_j, tf.constant(-1, dtype=v_j.dtype, name="pred_vj_const")),
                true_fn=lambda: self.build_tree(
                    theta_minus, r_minus,
                    u, v_j, j, epsilon, theta_last, r_0
                ),
                false_fn=lambda: self.build_tree(
                    theta_plus, r_plus,
                    u, v_j, j, epsilon, theta_last, r_0
                )
            )

            assign_ops = tf.cond(
                pred=tf.equal(v_j, tf.constant(-1, dtype=v_j.dtype, name="pred2_vj_const")),
                true_fn=lambda: [tf.assign(theta_minus, theta_minus_tree),
                                 tf.assign(r_minus, r_minus_tree)],
                false_fn=lambda: [tf.assign(theta_plus, theta_plus_tree),
                                  tf.assign(r_plus, r_plus_tree)],
            )

            with tf.control_dependencies(assign_ops):
                probability_bound = tf.minimum(
                    tf.constant(1., dtype=tf.float64, name="probbound_1_const"),
                    tf.divide(n_prime, n)
                )

                theta_m = tf.cond(
                    pred=tf.logical_and(
                        tf.equal(s_prime, tf.constant(1, dtype=s_prime.dtype)),
                        tf.less_equal(
                            tf.random_uniform(shape=(), maxval=1., dtype=tf.float64),
                            probability_bound
                        )
                    ),
                    true_fn=lambda: [tf.assign(theta_val, theta_prime_val) for theta_val, theta_prime_val in zip(params, theta_prime)],
                    false_fn=lambda: theta_last
                )

                s_new = tf.multiply(
                    s_prime * indicator(
                        tf.greater_equal((theta_plus - theta_minus) * r_minus, 0),
                    ),
                    indicator(
                        tf.greater_equal((theta_plus - theta_minus) * r_plus, 0)
                    )
                )

                return (
                    s_new, n + n_prime, j + 1,
                    theta_m,
                    theta_minus, r_minus,
                    theta_plus, r_plus,
                    r_0, u
                )

        a, *_ = tf.while_loop(
            cond=lambda s, *_: tf.equal(s, tf.constant(1, dtype=s.dtype)),
            body=loop_body,
            loop_vars=[s, n, j, params, params, r_0, params, r_0, r_0, u]
        )
        session.run(a)

        # XXX: Mass adaptation to estimate epsilon

    def leapfrog(self, theta, r, epsilon):
        # XXX: Currently limited to 1d values!
        old_costs = self.cost_fun(theta)
        old_gradient = tf.gradients(old_costs, theta)[0]
        r_bar = r + tf.divide(epsilon, tf.constant(2., dtype=epsilon.dtype)) * old_gradient

        theta_bar = theta + epsilon * r_bar
        new_costs = self.cost_fun(theta_bar)
        new_gradient = tf.gradients(new_costs, theta_bar)[0]
        r_bar = r_bar + tf.divide(epsilon, tf.constant(2., dtype=epsilon.dtype)) * new_gradient
        return theta_bar, r_bar

    def build_tree(self, theta, r, u, v, j, epsilon, theta_0, r_0):
        costs_theta_0 = self.cost_fun(theta_0)

        def base_case(theta, r, v, epsilon, u, r_0, costs_theta_0):
            theta_prime, r_prime = self.leapfrog(theta=theta, r=r, epsilon=tf.cast(v, dtype=epsilon.dtype) * epsilon)

            costs_theta_prime = self.cost_fun(theta_prime)

            n_prime = indicator(
                tf.less_equal(
                    u, tf.exp(costs_theta_prime - 0.5 * r_prime * r_prime)
                )
            )

            s_prime = indicator(
                tf.less(
                    u,
                    tf.exp(self.delta_max + costs_theta_prime - 0.5 * r_prime * r_prime)
                )
            )

            minimum_term = tf.minimum(
                tf.constant(1., dtype=costs_theta_prime.dtype, name="probbound2_const"),
                tf.exp(
                    costs_theta_prime - 0.5 * r_prime *
                    r_prime - costs_theta_0 + 0.5 * r_0 * r_0
                )
            )

            return (
                theta_prime, r_prime, theta_prime, r_prime,
                theta_prime, n_prime, s_prime, minimum_term, 1
            )

        def recursion(theta, r, u, v, j, epsilon, theta_0, r_0):
            (theta_minus, r_minus, theta_plus, r_plus,
             theta_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = self.build_tree(
                theta, r, u, v, j - 1, epsilon, theta_0, r_0
            )

            def true_fn_sprime_equals_1(theta_minus, r_minus,
                                        theta_plus, r_plus,
                                        u, v, j, epsilon, theta_0, r_0,
                                        alpha_prime, n_alpha_prime, n_prime):
                (theta_minus_tree, r_minus_tree,
                 theta_plus_tree, r_plus_tree,
                 theta_prime_prime, n_prime_prime,
                 s_prime_prime, alpha_prime_prime,
                 n_alpha_prime_prime) = tf.cond(
                    pred=tf.equal(v, tf.constant(-1, dtype=v.dtype, name="pred1_v_const")),
                    true_fn=lambda: self.build_tree(
                        theta_minus, r_minus, u, v, j - 1,
                        epsilon, theta_0, r_0
                    ),
                    false_fn=lambda: self.build_tree(
                        theta_plus, r_plus, u, v, j - 1,
                        epsilon, theta_0, r_0
                    )
                )

                theta_minus, r_minus = tf.cond(
                    pred=tf.equal(v, tf.constant(-1., dtype=v.dtype, name="pred2_v_const")),
                    true_fn=lambda: (theta_minus_tree, r_minus_tree),
                    false_fn=lambda: (theta_minus, r_minus)
                )
                theta_plus, r_plus = tf.cond(
                    pred=tf.equal(v, tf.constant(-1., dtype=v.dtype, name="pred3_v_const")),
                    true_fn=lambda: (theta_plus, r_plus),
                    false_fn=lambda: (theta_plus_tree, r_plus_tree),
                )

                theta_prime = tf.cond(
                    pred=tf.less_equal(
                        tf.random_uniform(shape=(), maxval=0., dtype=n_prime_prime.dtype),
                        tf.divide(n_prime_prime, n_prime + n_prime_prime),
                    ),
                    true_fn=lambda: theta_prime_prime,
                    false_fn=lambda: theta_prime
                )
                alpha_prime = alpha_prime + alpha_prime_prime
                n_alpha_prime = n_alpha_prime + n_alpha_prime_prime
                n_prime = n_prime + n_prime_prime

                s_prime = tf.multiply(
                    s_prime_prime * indicator(
                        tf.greater_equal(
                            (theta_plus - theta_minus) * r_minus,
                            tf.constant(0., dtype=theta_plus.dtype, name="gteq_const_0")
                        )
                    ), indicator(
                        tf.greater_equal(
                            (theta_plus - theta_minus) * r_plus,
                            tf.constant(0., dtype=theta_plus.dtype, name="gteq_const_1"))
                    )
                )

                return (
                    theta_minus, r_minus, theta_plus, r_plus,
                    theta_prime, n_prime, s_prime,
                    alpha_prime, n_alpha_prime
                )

            theta_minus, r_minus, theta_plus, r_plus,
            theta_prime, n_prime, s_prime,
            alpha_prime, n_alpha_prime = tf.cond(
                pred=lambda: tf.equal(s_prime, tf.constant(1., dtype=s_prime.dtype, name="s_prime_const")),
                true_fn=lambda: true_fn_sprime_equals_1(
                    theta_minus, r_minus,
                    theta_plus, r_plus,
                    u, v, j, epsilon, theta_0, r_0,
                    alpha_prime, n_alpha_prime, n_prime
                ),
                false_fn=lambda: (
                    theta_minus, r_minus, theta_plus, r_plus,
                    theta_prime, n_prime, s_prime,
                    alpha_prime, n_alpha_prime
                )
            )
            return (
                theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime,
                s_prime, alpha_prime, n_alpha_prime
            )
        (theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime,
         s_prime, alpha_prime, n_alpha_prime) = tf.cond(
            pred=tf.equal(j, tf.constant(0, dtype=j.dtype, name="0_const0")),
            true_fn=lambda: base_case(theta, r, v, epsilon, u, r_0, costs_theta_0),
            false_fn=lambda: recursion(theta, r, u, v, j, epsilon, theta_0, r_0)
        )
        return (
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime,
            s_prime, alpha_prime, n_alpha_prime
        )
