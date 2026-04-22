from conf import *
import torch
from benchmark.hf_models import load_mistral32_24b
from benchmark.baseline import run_baselines
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import BatchEncoding

mistral, tokenizer = load_mistral32_24b()

datasets = ["validation_lp_balanced", "validation_rp_balanced"]

prompt_formats = [False, True]


def preprocess_fn(text):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
    input_ids = torch.tensor([tokenized.tokens])
    attention_mask = torch.ones_like(input_ids)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask}), {}

def decode_fn(tokens, inputs):
    tokens = tokens[0]
    return tokenizer.decode(tokens)

run_baselines(mistral, "mistral32_24B", datasets, prompt_formats, preprocess_fn=preprocess_fn, decode_fn=decode_fn)
