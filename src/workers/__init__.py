from importlib import import_module

_EXPORTS = {
    "LLM_as_Jev": (".jev", "LLM_as_Jev"),
    "API_Jev": (".jev", "API_Jev"),
    "Laya": (".jev", "Laya"),
    "Kev": (".jev", "Kev"),
    "VLLMGenerator": (".llm_generator", "VLLMGenerator"),
    "APIGenerator": (".llm_generator", "APIGenerator"),
    "BM25Retriever": (".retriever", "BM25Retriever"),
    "BM25Index": (".retriever", "BM25Index"),
    "SQLiteBM25Retriever": (".retriever", "SQLiteBM25Retriever"),
    "Qwen3Reranker": (".reranker", "Qwen3Reranker"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _EXPORTS[name]
    module = import_module(module_name, __package__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
