from conf import *
import torch

from dataset.processor import pad, train_curriculum, special_tokens, prepare_ds_inference, process
from model.type_llama_no_ffn import ModelArgs, Transformer
from dataset.data_preprocessing import pad_collate
from dataset.eval import eval_model
from datasets import load_dataset
from torch.utils.data import DataLoader

from evaluation.model_repository import ModelRepository
from training.train_loop import load_weights_and_init

args = lambda x: ModelArgs(vocab_size=256, pad_token_id=pad, generated_type_indices=[0, 1], **x)
path = EXPERIMENTS_DIR + f"/models/type_llama_no_ffn/"
repo = ModelRepository(base_path=path, model_class=Transformer, loader_func=load_weights_and_init, args_builder=args)


def base_eval(model, is_cot=False, distribution="validation_rp_balanced", eval_f=eval_model):
    collate_fn = lambda x: pad_collate(x, padding=pad)
    full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to("cuda")
    model.eval()

    inference_ids = train_curriculum(full_dataset_dict[distribution])
    ds = DataLoader(prepare_ds_inference(process(inference_ids), is_cot=is_cot), collate_fn=collate_fn)
    with torch.no_grad():
        eval = eval_f(model, ds, vocabulary=special_tokens, answer_position=-1, is_cot=is_cot)

    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to("cpu")
    return eval


# eval any model on any distribution
entry = repo.get_entry('rp', 'corrective', 'r2', 'bidir')
distributions = ["validation_rp_balanced"]
for distribution in distributions:
    for is_cot in [False, True]:
        print(f"Evaluating {entry.pretty_name} on {distribution} is_cot={is_cot}. model={entry.model.params}")
        accuracies = base_eval(entry.model, distribution=distribution, is_cot=is_cot)
        decimal_accuracies = {layer: count / 1000 for layer, count in accuracies.items()}
        print((distribution, is_cot, *entry.tags), decimal_accuracies)
