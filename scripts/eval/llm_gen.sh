#!/bin/bash

LIMIT=1000

MODELS=(
    '/home/yxfeng/models/Qwen2.5/Qwen2.5-0.5B-Instruct'
    '/home/yxfeng/models/Qwen2.5/Qwen2.5-1.5B-Instruct'
    '/home/yxfeng/models/Qwen2.5/Qwen2.5-3B-Instruct'
    '/home/yxfeng/models/Qwen2.5/Qwen2.5-7B-Instruct'
    '/home/yxfeng/models/Qwen2.5/Qwen2.5-14B-Instruct'
    '/home/yxfeng/models/Qwen2.5/Qwen2.5-32B-Instruct'
    # '/root/autodl-tmp/models/Qwen3-0.6B'  # not in /home/yxfeng/models
    # '/root/autodl-tmp/models/Qwen3-1.7B'  # not in /home/yxfeng/models
    '/home/yxfeng/models/Qwen3/Qwen3-4B'
    '/home/yxfeng/models/Qwen3/Qwen3-8B'
    '/home/yxfeng/models/Qwen3/Qwen3-14B'
    '/home/yxfeng/models/Qwen3.8/Qwen3.8-27B'
    # '/root/autodl-tmp/models/Qwen3.5-4B'  # not in /home/yxfeng/models
    '/home/yxfeng/models/Llama-3.2/Llama-3.2-3B'
    # '/home/yxfeng/models/mistral/Mistral-7B-Instruct-v0.2'
)

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "开始运行模型: $MODEL"
    echo "========================================"

    /home/yxfeng/miniconda3/envs/www27/bin/python /home/yxfeng/www27/eval/llm_gen.py \
        --model "$MODEL" \
        --limit $LIMIT \
        --output_dir "/data3/yxfeng/www27/llm_gen_only/$(basename "$MODEL")"
done

echo "所有模型运行完毕！"