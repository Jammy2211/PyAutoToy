from functools import wraps


def assert_lengths_match(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self, first, second = args + tuple(kwargs.values())
        if len(first) != len(second):
            raise AssertionError(
                f"Length of lists passed to {func.__name__} do not match"
            )
        return func(self, first, second)

    return wrapper