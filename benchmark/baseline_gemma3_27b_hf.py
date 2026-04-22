from conf import *
import torch
from benchmark.hf_models import load_gemma3_27B
from benchmark.baseline import run_baselines

gemma, processor = load_gemma3_27B()

datasets = ["validation_lp_balanced", "validation_rp_balanced"]

prompt_formats = [False, True]


def preprocess_fn(text):
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    tokens = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dtype=torch.bfloat16)
    return tokens, {}


def decode_fn(tokens, inputs):
    tokens = tokens[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(tokens, skip_special_tokens=False)[0]


run_baselines(gemma, "gemma3_27b", datasets, prompt_formats, preprocess_fn=preprocess_fn, decode_fn=decode_fn)
