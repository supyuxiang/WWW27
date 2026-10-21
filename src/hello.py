from transformers import AutoTokenizer

model_path='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
tok = AutoTokenizer.from_pretrained(model_path)

texts = ['hello, how are you?', 'i am fine, thank you!']

enc = tok(
    texts,
    padding=False
)
breakpoint()