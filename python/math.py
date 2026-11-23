"""Public math entry point for ViDo.

The implementation lives in vido_math.py so a file named `math.py` does not
shadow Python's standard library module when the server imports torch.
"""

from vido_math import (  # noqa: F401
    evaluate_basic,
    handle_advanced,
    handle_query,
)
