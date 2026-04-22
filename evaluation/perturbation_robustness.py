from conf import *

import torch
from dataset.processor import pad
from model.type_llama_no_ffn import ModelArgs, Transformer
from base_eval import base_eval
from evaluation.model_repository import ModelRepository
from training.train_loop import load_weights_and_init

args = lambda x: ModelArgs(vocab_size=256, pad_token_id=pad, generated_type_indices=[0, 1], **x)
path = EXPERIMENTS_DIR + f"/models/type_llama_no_ffn/"
repo = ModelRepository(base_path=path, model_class=Transformer, loader_func=load_weights_and_init, args_builder=args)

for evaluation in [{"rp", "corrective", "r2", "bidir", "scaling", "layers=64"}]:
    entry = repo.get_entry(*evaluation)

    STD = 3.0
    ORDER = "random"
    with torch.no_grad():
        embed_tensor = entry.model.type_embeddings
        original_embeddings = embed_tensor.clone()
        mean = embed_tensor.mean(dim=-1, keepdim=True)
        std = embed_tensor.std(dim=-1, keepdim=True)
        z_scores = (embed_tensor - mean) / (std + 1e-9)
        spike_mask = torch.abs(z_scores) > STD
        embed_tensor.mul_(spike_mask)
        kept_dims = spike_mask.sum(dim=-1)
        print(f"Active dimensions per type: {spike_mask.sum(dim=-1).tolist()}")

        # embed_tensor[1, 98] = 0.0 # fact@98
        # embed_tensor[1, 104] = 0.0 # fact@104
        # embed_tensor[1, 230] = 0.0 # fact@230
        # embed_tensor[1, 241] = 0.0 # fact@241
        # embed_tensor[1, :] = 0.0 # fact@

    distributions = ["validation_rp_balanced"]
    analysis_results = {}
    for distribution in distributions:
        for is_cot in [False]:
            print(f"Evaluating {entry.pretty_name} on {distribution} is_cot={is_cot}. model={entry.model.params}")
            accuracies = base_eval(entry.model, distribution=distribution, is_cot=is_cot, solver_order=ORDER)[0]
            decimal_accuracies = {layer: count / 1000 for layer, count in accuracies.items()}
            analysis_results[(distribution, is_cot, *entry.tags)] = decimal_accuracies
    print(analysis_results)

    with torch.no_grad():
        entry.model.type_embeddings.copy_(original_embeddings)

# layers=64
# default: 7000
# std=3.0: 6997 types: [0, 4, 4, 4, 4, 4, 0, 0, 3, 0]
# std=4.0: 6993 types: [0, 4, 3, 2, 4, 4, 0, 0, 2, 0]
# std=5.0: 6939 types: [0, 4, 1, 2, 4, 3, 0, 0, 2, 0]

# std=3.0 ablation
# fact@  totl: {depth: correct}

# facts rules: 6997
# 98     6996: {0: 996, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 1000}
# 104    6962: {0: 968, 1: 994, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 1000}
# 230    6305: {0: 931, 1: 920, 2: 913, 3: 910, 4: 892, 5: 875, 6: 864}
# 241    6995: {0: 995, 1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000, 6: 1000}
# all    6510: {0: 936, 1: 954, 2: 948, 3: 935, 4: 932, 5: 902, 6: 903}

# rules facts: 6651:   {0: 973, 1: 965, 2: 937, 3: 935, 4: 947, 5: 944, 6: 950}
# 98     6234: {0: 955, 1: 902, 2: 881, 3: 868, 4: 897, 5: 873, 6: 858}
# 104    6414: {0: 946, 1: 959, 2: 933, 3: 903, 4: 915, 5: 892, 6: 866}
# 230    3663: {0: 620, 1: 528, 2: 506, 3: 500, 4: 502, 5: 505, 6: 502}
# 241    6461: {0: 963, 1: 936, 2: 916, 3: 903, 4: 927, 5: 910, 6: 906}
# all    4115: {0: 703, 1: 630, 2: 581, 3: 569, 4: 563, 5: 532, 6: 537}

# random     : 6909:   {0: 995, 1: 994, 2: 983, 3: 984, 4: 984, 5: 986, 6: 983}
# 230    3906: {0: 707, 1: 574, 2: 547, 3: 529, 4: 519, 5: 516, 6: 514}
# all    4096: {0: 723, 1: 627, 2: 591, 3: 550, 4: 541, 5: 533, 6: 531}
