'''JARV: Judgment-guided Adaptive Retrieval and Verification.

Training-free fact-checking with a bounded gate -> retrieve -> gate loop.
Bounded decisions use LLM-as-Jev candidate suffix likelihoods. Query
construction is ordinary generation; retrieval is a local BM25 index.
'''

__version__ = '0.2.0'
