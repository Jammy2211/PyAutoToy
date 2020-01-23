from functools import wraps

LOWER_LIMIT = 0
UPPER_LIMIT = 20
NUMBER_OF_POINTS = 400


def pdf(observable):
    return observable.pdf(
        LOWER_LIMIT,
        UPPER_LIMIT,
        NUMBER_OF_POINTS
    )


def assert_lengths_match(func):
    """
    Decorator for methods that take two lists that asserts the lists are the same length.

    Parameters
    ----------
    func
        A method that takes two lists

    Returns
    -------
    The same method but with a check that raises an AssertionError should the list lengths
    not match.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        self, first, second = args + tuple(kwargs.values())
        if len(first) != len(second):
            raise AssertionError(
                f"Length of lists passed to {func.__name__} do not match"
            )
        return func(self, first, second)

    return wrapper
