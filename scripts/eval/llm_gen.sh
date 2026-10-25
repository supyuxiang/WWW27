#!/bin/bash

LIMIT=1000

MODELS=(
    '/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct'
    '/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
    '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'
    '/root/autodl-tmp/models/Qwen2.5-14B-Instruct'
    # '/root/autodl-tmp/models/Qwen2.5-32B-Instruct'
    '/root/autodl-tmp/models/Qwen3-0.6B'
    '/root/autodl-tmp/models/Qwen3-1.7B'
    '/root/autodl-tmp/models/Qwen3-4B'
    '/root/autodl-tmp/models/Qwen3-8B'
    '/root/autodl-tmp/models/Qwen3-14B'
    '/root/autodl-tmp/models/Qwen3.5-4B'
    '/root/autodl-tmp/models/Llama-3.2-3B-Instruct'
    '/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3'
)

for MODEL in "${MODELS[@]}"; do
    echo "========================================"
    echo "开始运行模型: $MODEL"
    echo "========================================"

    /root/autodl-tmp/miniconda3/envs/www27/bin/python /root/yxfeng/www27/eval/llm_gen.py \
        --model "$MODEL" \
        --limit $LIMIT \
        --output_dir "/root/autodl-tmp/www27/llm_gen_only/$(basename "$MODEL")"
done

echo "所有模型运行完毕！"