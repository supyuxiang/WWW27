# Doffy (Decoding-free Fact-Verifying)

---

## 代码结构

```text
www27/
├── src/
│   ├── workers/
│   │   ├── __init__.py        # 导出下面这些类
│   │   ├── jev.py             # LLM_as_Jev：对候选 verdict 打 token log-prob
│   │   ├── llm_generator.py   # VLLMGenerator / APIGenerator：本地 vLLM 与 DeepSeek API
│   │   ├── retriever.py       # BM25Retriever（内存）/ SQLiteBM25Retriever（FTS5）
│   │   └── reranker.py        # Qwen3Reranker：按 query 对 passage 打 yes/no 概率
│   ├── prompts.py             # 判定、few-shot、检索门控与 aspect 提示
│   ├── utils.py               # 数据加载、Accuracy / Macro-F1、答案解析
│   ├── config.py              # Document、Passage 与实验配置
│   └── aspects.py             # 事实核查切面定义
├── eval/                      # 评测入口
├── scripts/eval/              # 多模型循环脚本
├── data/preprocess.py         # 把 benchmark 写成统一 jsonl
├── demo_bm25.py               # fastembed BM25 点积示例
└── third_party/Math-Verify/   # 从生成文本里抽取 \boxed{} 答案
```

评测脚本通过 `from src.workers import ...` 使用这些类。

评测入口：

| 脚本 | 作用 | 启动 |
|---|---|---|
| `eval/jev_only.py` | 金证据上的 JEV 判定 | `scripts/eval/jev_only.sh` |
| `eval/llm_gen.py` | 金证据上的生成式判定，与 JEV 对比 | `scripts/eval/llm_gen.sh` |
| `eval/jev_retrieval_once.py` | 先 BM25 检索一次，再用 JEV 判定 | `scripts/eval/jev_retrieval_once.sh` |
| `eval/eval_main.py` | Doffy 主流程：检索门控、aspect、重排、JEV 判定 | 直接运行该脚本 |

预处理后的 claim 为 `id`、`claim`、`label`、`evidence`，语料为 `doc_id`、`title`、`text`。加载与指标在 `src/utils.py`。

---

## RQ1：验证 LLM-as-Jev 具有 training-free 的 fact-checking 能力  
**设计离线检索实验去验证 Jev + Retrieval Once 的能力**

### Models
- Qwen2.5-0.5/1.5/3/7/14/32B-Instruct  
- Qwen3-0.6/1.7/4/8/14B  
- Qwen3.5-4B  
- Qwen3.8-27B  
- Llama3.2-3B-Instruct  
- Mistral-7B-Instruct-v0.3  

### Settings（与 Generation 对比，含闭源模型 ds-R1 / v4flash / v4pro, gpt-4o）
- Zero-shot  
- Few-shot  
- + Reasoning?  
- + Fine-tuning?  

### Benchmarks
- **SciFact** (3-way, dev 集 300 条)  
- **FEVER** (3-way, 随机抽 1k 条测试)  
- **AVeriTeC** (4-way, 含多跳 + 时间敏感，较难，随机抽 300 条测试)  
- **HealthVer** (3-way, 随机抽 1k 条)  

### Evaluation
- Accuracy  
- Macro-F1  
- 各类别 Precision / Recall / F1  

### Case Study
- Jev 对 / Gen 错  
- Jev 错 / Gen 对  
- NEI 边界  

### 时间
2 day (+ code review)

---

## RQ2：我们的框架 —— Doffy（Decoding-free Fact-Verifying）的设计，Main Result

### 时间
2 day

### Models
*（待补充）*

---

## RQ3：Ablation

### Models
*（待补充）*

### 时间
5 h

---

## RQ4：Efficiency

### Models
*（待补充）*

### 时间
3 h