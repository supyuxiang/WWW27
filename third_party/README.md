# Vendored dependencies

## Math-Verify

Source: https://github.com/huggingface/Math-Verify

Vendored commit: `ba3d3aaff23b3f4cac7a14672b4f6e293d97c98b`

This copy is intentionally stored in-tree so that boxed-answer extraction and
LaTeX normalization can be modified locally. `eval/llm_gen_only.py` puts
`third_party/Math-Verify/src` ahead of site-packages on `sys.path`.

Runtime dependency: `latex2sympy2_extended==1.11.0`, installed through the
`math-verify[antlr4_9_3]` requirement.
