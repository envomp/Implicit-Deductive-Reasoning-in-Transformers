from conf import *
from benchmark.hf_models import load_nemotron3_30B
from benchmark.baseline import run_baselines

nemotron, tokenizer = load_nemotron3_30B()

datasets = ["validation_lp_balanced", "validation_rp_balanced"]

prompt_formats = [True, False]

def preprocess_fn(text):
    chat = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, enable_thinking="step-by-step" in text, add_generation_prompt=True)]
    return tokenizer(chat, return_tensors="pt"), {"do_sample": False, "num_beams": 1}

def decode_fn(tokens, inputs):
    tokens = tokens[:, inputs['input_ids'].shape[1]:]
    return tokenizer.batch_decode(tokens, skip_special_tokens=False)[0]

run_baselines(nemotron, "nemotron3_30B", datasets, prompt_formats, preprocess_fn=preprocess_fn, decode_fn=decode_fn)
