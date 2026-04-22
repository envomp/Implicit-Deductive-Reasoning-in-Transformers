from conf import *
from benchmark.hf_models import load_qwen3_30B
from benchmark.baseline import run_baselines

qwen, tokenizer = load_qwen3_30B()

datasets = ["validation_lp_balanced", "validation_rp_balanced"]

prompt_formats = [False, True]


def preprocess_fn(text):
    chat = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)]
    return tokenizer(chat, return_tensors="pt"), {}

def decode_fn(tokens, inputs):
    tokens = tokens[:, inputs['input_ids'].shape[1]:]
    return tokenizer.batch_decode(tokens, skip_special_tokens=False)[0]

run_baselines(qwen, "qwen3_30B", datasets, prompt_formats, preprocess_fn=preprocess_fn, decode_fn=decode_fn)
