import pytest
import sympy
from unittest.mock import patch

from math_verify.grader import verify


def test_verify_raise_on_error():
    """
    When the internal comparison raises an error, `verify` should:

    1. return False if `raise_on_error=False`.
    2. propagate the original exception if `raise_on_error=True`.
    """
    with patch("math_verify.grader.sympy_expr_eq", side_effect=ValueError("boom")):
        # Should silently fail and return False
        assert not verify(sympy.Integer(1), sympy.Integer(1), raise_on_error=False)

        # Should re-raise the underlying error
        with pytest.raises(ValueError, match="boom"):
            verify(sympy.Integer(1), sympy.Integer(1), raise_on_error=True) 