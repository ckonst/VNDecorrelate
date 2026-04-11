import time
from functools import wraps


def timed(repititions: int = 5):
    """
    A decorator to measure the execution time of a function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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
