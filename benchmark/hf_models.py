import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoTokenizer, AutoConfig


def inference(model, inputs, max_tokens=1000, kwargs={}):
    inputs = inputs.to(model.device)
    generation_args = {"max_new_tokens": max_tokens, "do_sample": False, **kwargs}
    return model.generate(**inputs, **generation_args)


def load_gemma3_4B():
    from transformers import Gemma3ForConditionalGeneration

    name = "google/gemma-3-4b-it"
    revision = "093f9f388b31de276ce2de164bdc2081324b9767"

    gemma = Gemma3ForConditionalGeneration.from_pretrained(name, revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
    for param in gemma.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(name, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in gemma.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return gemma, processor


def load_gemma3_27B():
    from transformers import Gemma3ForConditionalGeneration

    name = "google/gemma-3-27b-it"
    revision = "005ad3404e59d6023443cb575daa05336842228a"

    gemma = Gemma3ForConditionalGeneration.from_pretrained(name, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    for param in gemma.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(name, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in gemma.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return gemma, processor


def load_nemotron3_30B():
    name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    revision = "fcd023f36ea64638f74dab1781352f236ccf6c51"

    nemotron = AutoModelForCausalLM.from_pretrained(name, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    for param in nemotron.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in nemotron.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return nemotron, tokenizer


def load_qwen3_30B():
    name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    revision = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"

    qwen = AutoModelForCausalLM.from_pretrained(name, revision=revision, trust_remote_code=True, torch_dtype="auto", device_map="auto")
    for param in qwen.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in qwen.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return qwen, tokenizer


def load_granite4_32b():
    name = "ibm-granite/granite-4.0-h-small"
    revision = "b8c0982bab7fde4eb48110f5a069527c008fab39"

    granite4 = AutoModelForCausalLM.from_pretrained(name, revision=revision, trust_remote_code=True, torch_dtype="auto", device_map="auto")
    for param in granite4.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in granite4.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return granite4, tokenizer


def load_mistral32_24b():
    from transformers import Mistral3ForConditionalGeneration
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    name = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    revision = "cb17b97769b0305ddc717ede4a4ef6fd54ef8371"

    mistral = Mistral3ForConditionalGeneration.from_pretrained(name, revision=revision, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    for param in mistral.parameters():
        param.requires_grad = False

    tokenizer = MistralTokenizer.from_hf_hub(name, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in mistral.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return mistral, tokenizer
