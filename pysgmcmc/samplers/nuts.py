import tensorflow as tf

from pysgmcmc.samplers.base_classes import MCMCSampler
from pysgmcmc.tensor_utils import (
    vectorize, unvectorize, safe_divide, safe_sqrt,
)
from pysgmcmc.stepsize_schedules import ConstantStepsizeSchedule


class NUTSSampler(MCMCSampler):
    def __init__(self, params, cost_fun, batch_generator=None,
                 stepsize_schedule=ConstantStepsizeSchedule(0.01), seed=None,
                 session=tf.get_default_session(), dtype=tf.float64):
        pass


