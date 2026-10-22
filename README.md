
# RQ1：验证LLM-as-Jev具有training-free的fact-checking能力———设计离线检索实验去验证jev+retrieval once的能力
## (1) Models: Qwen2.5-0.5/1.5/3/7/14/32B-Instruct; Qwen3-0.6/1.7/4/8/14B; Qwen3.5-4B; Qwen3.8-27B; Llama3.2-3B-Instruct; Mistral-7B-Instruct-v0.3
## (2) Settings (compared with generation，含闭源模型): 
    zero-shot
    few-shot
    + reasoning？
    + fine-tuning?
## (3) Benchmarks
    SciFact (3-way, dev集300条)

    FEVER (3-way, 随机抽1k条测试)

    AVeriTeC (4-way, 含多跳 + 时间敏感，较难,随机抽300条测试)

    Climate-FEVER (3-way, 随机抽500条)

    HealthVer (3-way, 随机抽1k条)

## (4) Evaluaton
    Accuracy;
    Macro-F1;
    各类别 Precision/Recall/F1

## (4) Case study
    Jev 对 / Gen 错.
    Jev 错 / Gen 对.
    NEI边界.

## (5) 时间：1 day ( + code review)



# RQ2：我们框架，Doffy（Decoding-free fact-verifying）的设计，main result
## 时间 2 day
## Models

# RQ3: ablation
## Models
##  5h

# RQ4: efficiency
## Models
## 3h