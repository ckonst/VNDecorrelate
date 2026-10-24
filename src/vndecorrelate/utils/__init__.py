"""Utility functions for signal processing, plotting, and timing."""

import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec('P')
R = TypeVar('R')


def timed(repititions: int = 5) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Measure and print a function's average execution time.

    Parameters
    ----------
    repititions : int
        Number of times to call the decorated function. The parameter name is
        retained for backward compatibility.

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        Decorator that returns the original result after timing repeated calls.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Wrap a callable with repeated timing.

        Parameters
        ----------
        func : Callable[P, R]
            Function to time.

        Returns
        -------
        Callable[P, R]
            Timed wrapper for ``func``.
        """

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            """Call the wrapped function repeatedly and return its last result.

            Parameters
            ----------
            *args : P.args
                Positional arguments forwarded to the wrapped function.
            **kwargs : P.kwargs
                Keyword arguments forwarded to the wrapped function.

            Returns
            -------
            R
                Result from the final wrapped call.
            """
            execution_time = 0
            for _ in range(repititions):
                start_time = time.perf_counter()  # Use perf_counter for precise timing
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_time += end_time - start_time
            print(
                f"Function '{func.__name__}' executed {repititions} times averaging {(execution_time / repititions) * 1000:.4f} milliseconds."
            )
            return result

        return wrapper

    return decorator
