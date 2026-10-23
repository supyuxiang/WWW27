# Doffy (Decoding-free Fact-Verifying)

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
1 day (+ code review)

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