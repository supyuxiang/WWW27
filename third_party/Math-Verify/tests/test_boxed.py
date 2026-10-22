import pytest

from tests.test_all import compare_strings

@pytest.mark.parametrize(
    "gold, pred, expected",
    [
        (r"$\\boxed{35 cm} ++++++ \\boxed{4}$", r"${35 cm,4}$", 1),
        (
            r" the answer should be \\boxed{004}.\n\nBut let me check again: the sum is 2,552,004. The last three digits are 004. Therefore, when divided by 1000, the remainder is 004. So, \\boxed{004} is the correct answer.\n\n**Final Answer**\n\\boxed{004}",
            r"$004$",
            1,
        ),
        (
            r"SoHi YES.\n\n could answer therefore\\boxed{840}.,but let me put this after explain.\n\n**Final Answer**\n\\boxed{840}",
            r"$840$",
            1,
        ),
    ],
)
def test_boxed(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex", "expr"]) == expected
