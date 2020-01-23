from functools import wraps


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
