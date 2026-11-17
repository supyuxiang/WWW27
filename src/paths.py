from pathlib import Path

ROOT = Path("/data3/yxfeng/www27")
PREPROCESSED = ROOT / "preprocessed"
CORPUS = PREPROCESSED / "corpus"
CLAIMS = PREPROCESSED / "claims"
INDEX = ROOT / "index"
RESULTS = ROOT / "results"
RAW = ROOT / "raw"
LOGS = ROOT / "logs"

EMBED_MODEL = Path("/home/yxfeng/models/Qwen3/Qwen3-Embedding-4B")
JUDGE_MODEL = Path("/home/yxfeng/models/Qwen3/Qwen3-8B")
RERANK_MODEL = Path("/home/yxfeng/models/Qwen3/Qwen3-Reranker-4B")
