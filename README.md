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
│   │   └── reranker.py        # Qwen3Reranker：按 claim 和 aspect 对 passage 打 yes/no 概率
│   ├── prompts.py             # 判定、few-shot、检索门控与 aspect 提示
│   ├── utils.py               # 数据加载、Accuracy / Macro-F1、各类别 P/R/F1、答案解析
│   ├── config.py              # Document、Passage 与实验配置
│   └── aspects.py             # 五个事实核查切面
├── eval/
│   ├── jev_only.py            # 证据 + JEV
│   ├── llm_gen.py             # 证据 + 生成式判定
│   ├── jev_retrieval_once.py  # BM25 top 500 + rerank top 2 + JEV
│   └── eval_main.py           # Doffy 主循环
├── scripts/eval/
│   ├── jev_only.sh
│   ├── llm_gen.sh
│   ├── eval_gen_cot.sh        # 生成式判定，扫 max_tokens；Qwen3 打开 thinking
│   ├── jev_retrieval_once.sh
│   └── eval_main.sh
├── baselines/                 # FIRE、GProofT 等已发表基线的复现
├── data/preprocess.py         # 把 benchmark 写成统一 jsonl
├── demo_bm25.py               # fastembed BM25 点积示例
└── third_party/Math-Verify/   # 从生成文本里抽取 \boxed{} 答案
```

评测入口：

| 脚本 | 作用 | 启动 |
|---|---|---|
| `eval/jev_only.py` | 金证据上的 JEV 判定 | `scripts/eval/jev_only.sh` |
| `eval/llm_gen.py` | 金证据上的生成式判定，与 JEV 对比 | `scripts/eval/llm_gen.sh`，CoT 扫描用 `scripts/eval/eval_gen_cot.sh` |
| `eval/jev_retrieval_once.py` | 先 BM25 检索一次，再用 JEV 判定 | `scripts/eval/jev_retrieval_once.sh` |
| `eval/eval_main.py` | Doffy 主流程：检索门控、aspect、重排、JEV 判定 | `scripts/eval/eval_main.sh` |

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

## Methodology：Doffy

Doffy 把事实核查里的每一步选择都做成 **decoding-free** 的判别：不自回归生成答案，只对候选 token 打一次前缀 log-prob。

### LLM-as-Jev

判定、检索门控和 aspect 选择共用同一个打分器 `LLM_as_Jev.score_options`。

1. 用 chat template 拼好 system 和 user，并在助手开头预填 `Best answer: [`。
2. 每个选项是单个 token，编号为 `"1"`、`"2"`、……。
3. 前缀只前向一次。最后一个位置的 logits 就是下一个 token 的分布。
4. 读出各选项 token 的 log-prob，再 softmax。概率最高的选项就是决定。

因此门控、选 aspect、最终 verdict 都不产生一段自由文本。生成式对照实验在 `eval/llm_gen.py`，它才逐 token 解码，并把最终答案放进 `\boxed{}`。

### 检索循环

对每条 claim，证据从空开始，最多检索 `search_cap` 轮（默认 3）。每一轮在最终判定之前先问模型：

1. **要不要检索。** `prompt4jev_whether_retrieval` 的两个选项是 `retrieval` 和 `skip`。选到 skip 就停，用当前证据做最终判定。第一轮没有证据时也会先问。
2. **查哪一面。** 选到 retrieval 后，`prompt4jev_what_aspect` 对五个切面打分，在还没查过的切面里取得分最高的一个。五个切面定义在 `src/aspects.py`：`event_context`、`temporal_scope`、`quantities`、`attribution_actions`、`causality`。
3. **取证据。** 用 claim 做 BM25，取 top 500，并排除已经选过的文档。FEVER 和 AVeriTeC 走 SQLite FTS5 索引；SciFact 和 HealthVer 把语料载入内存。再用 `Qwen3-Reranker-4B` 按该 aspect 重排，留下 top 2，并进已有证据。

循环结束后，`prompt4jev_judge` 把 claim、合并后的证据和 few-shot 一起交给 JEV，选出该数据集的 verdict。

`eval/jev_retrieval_once.py` 是去掉门控和 aspect 循环的对照：每条 claim 只做一次 BM25 top 500 和 rerank top 2，然后直接判定。

## RQ2：我们的框架 —— Doffy（Decoding-free Fact-Verifying）的设计，Main Result

主实验入口是 `scripts/eval/eval_main.sh`。默认设置是 BM25 top 500、rerank top 2、最多 3 轮检索。指标与 RQ1 相同：Accuracy、Macro-F1、各类别 Precision / Recall / F1。

### 时间
2 day

### Models
与 RQ1 的开源模型列表相同。重排器固定为 `Qwen3-Reranker-4B`。

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