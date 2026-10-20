from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-3B-Instruct',trust_remote_code=True)
true_id = tokenizer.convert_tokens_to_ids("yes")

breakpoint()
