import sympy
from math_verify import parser

def test_gamma_as_function():
    latex_str = r"\(\Gamma(\frac{\pi}{12})\)"
    [parsed_str] = parser.parse(latex_str, fallback_mode="no_fallback")
    expected = sympy.gamma(sympy.pi/12)
    assert sympy.simplify(parsed_str - expected) == 0

def test_gamma_as_constant():
    latex_str = r"\(\Gamma * (\frac{\pi}{12})\)"
    [parsed_str] = parser.parse(latex_str, fallback_mode="no_fallback")
    expected = sympy.EulerGamma * sympy.pi/12
    assert sympy.simplify(parsed_str - expected) == 0
