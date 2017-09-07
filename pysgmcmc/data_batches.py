import numpy as np
import logging


def generate_batches(X, y, x_placeholder, y_placeholder, batch_size=20, seed=None):
    """ Infinite generator of random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    X: np.ndarray (N, D)
        Training data points/features

    y : np.ndarray (N, 1)
        Training data labels

    x_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `X`.

    y_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `y`.

    batch_size : int, optional
        Number of datapoints to put into a batch.

    seed: int, optional
        Random seed to use during batch generation.
        Defaults to `None`.

    Yields
    -------
    batch_dict : dict
        A dictionary that maps `x_placeholder` and `y_placeholder`
        to `batch_size` sized minibatches of data (numpy.ndarrays)
        from the dataset `X`, `y`.

    Examples
    -------
    Simple batch extraction example:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 100, 3  # 100 datapoints with 3 features each
    >>> X = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> X.shape, y.shape
    ((100, 3), (100,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(X, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((20, 3), (20, 1))

    Batch extraction resizes batch size if dataset is too small:

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> N, D = 10, 3  # 10 datapoints with 3 features each
    >>> X = np.asarray([np.random.uniform(-10, 10, D) for _ in range(N)])
    >>> y = np.asarray([np.random.choice([0., 1.]) for _ in range(N)])
    >>> X.shape, y.shape
    ((10, 3), (10,))
    >>> x_placeholder, y_placeholder = tf.placeholder(dtype=tf.float64), tf.placeholder(dtype=tf.float64)
    >>> batch_size = 20
    >>> gen = generate_batches(X, y, x_placeholder, y_placeholder, batch_size)
    >>> batch_dict = next(gen)  # extract a batch
    >>> set(batch_dict.keys()) == set((x_placeholder, y_placeholder))
    True
    >>> batch_dict[x_placeholder].shape, batch_dict[y_placeholder].shape
    ((10, 3), (10, 1))

    In this case, the batches contain exactly all datapoints:

    >>> np.allclose(batch_dict[x_placeholder], X), np.allclose(batch_dict[y_placeholder].reshape(N,), y)
    (True, True)

    """

    # Sanitize inputs
    assert(isinstance(batch_size, int)), "generate_batches: batch size must be an integer."
    assert(batch_size > 0), "generate_batches: batch size must be greater than zero."

    assert(seed is None or isinstance(seed, int)), "generate_batches: seed must be an integer or `None`"

    assert(seed is None or (0 <= seed <= 2 ** 32 - 1))

    assert(y.shape[0] == X.shape[0]), "Not exactly one label per datapoint!"

    n_examples = X.shape[0]

    if seed is not None:
        np.random.seed(seed)

    # Check if we have enough data points to form a minibatch
    # otherwise set the batchsize equal to the number of input points
    initial_batch_size = batch_size
    # print(batch_size)
    batch_size = min(initial_batch_size, n_examples)
    # print(batch_size)

    if initial_batch_size != batch_size:
        logging.error("Not enough datapoints to form a minibatch. "
                      "Batchsize was set to {}".format(batch_size))

    while True:
        # `np.random.randint` is end-exclusive => for n_examples == batch_size, start == 0 holds
        start = np.random.randint(0, (n_examples - batch_size + 1))

        minibatch_x = X[start:start + batch_size]
        minibatch_y = y[start:start + batch_size, None]

        feed_dict = {
            x_placeholder: minibatch_x,
            y_placeholder: minibatch_y.reshape(-1, 1)
        }
        yield feed_dict


def generate_shuffled_batches(X, y, x_placeholder, y_placeholder,
                              batch_size=20, seed=None):

    """ Infinite generator of shuffled random minibatches for a dataset.

        For general reference on (infinite) generators, see:
        https://www.python.org/dev/peps/pep-0255/

    Parameters
    ----------
    X: np.ndarray (N, D)
        Training data points/features

    y : np.ndarray (N, 1)
        Training data labels

    x_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `X`.

    y_placeholder : tensorflow.placeholder
        Placeholder for batches of data from `y`.

    batch_size : int, optional
        Number of datapoints to put into a batch.

    seed: int, optional
        Random seed to use during batch generation (and for shuffling!).
        Defaults to `None`.

    Yields
    -------
    batch_dict: dict
        A dictionary that maps `x_placeholder` and `y_placeholder`
        to `batch_size` sized minibatches of data (numpy.ndarrays)
        from the dataset `X`, `y`.

    Examples
    -------
    TODO Add some test examples that demonstrate shuffling
    """

    # always use a seed in order to shuffle x and y in the same way
    if seed is None:
        seed = np.random.randint(1, 100000)
    rng_x, rng_y = np.random.RandomState()
    rng_x.seed(seed)
    rng_y.seed(seed)

    for batch in generate_batches(X, y, x_placeholder, y_placeholder, batch_size, seed):
        # shuffles x and y in the same way
        rng_x.shuffle(batch[x_placeholder])
        rng_y.shuffle(batch[y_placeholder])
        yield batch
