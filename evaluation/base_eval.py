import torch
from dataset.processor import pad, train_curriculum, special_tokens, prepare_ds_inference, process
from dataset.data_preprocessing import pad_collate
from dataset.eval import eval_model
from datasets import load_dataset
from torch.utils.data import DataLoader

def base_eval(model, is_cot=False, distribution=None, distributions=["lp", "lp_star", "rp"],
              eval_f=eval_model, pred_count=30, solver_order="facts_rules"):
    collate_fn = lambda x: pad_collate(x, padding=pad)
    full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to("cuda")
    model.eval()
    evals = []

    if distribution:
        inference_ids = train_curriculum(full_dataset_dict[distribution])
        processed_ids = process(inference_ids, solver_order=solver_order)
        ds = DataLoader(prepare_ds_inference(processed_ids, is_cot=is_cot), collate_fn=collate_fn)
        with torch.no_grad():
            evals.append(eval_f(model, ds, vocabulary=special_tokens, answer_position=-1, is_cot=is_cot))
    else:
        for distribution in distributions:
            inference_ids = train_curriculum(full_dataset_dict[f"validation_{distribution.lower()}_balanced_deep_{pred_count}_pred"])
            processed_ids = process(inference_ids, solver_order=solver_order)
            ds = DataLoader(prepare_ds_inference(processed_ids, is_cot=is_cot), collate_fn=collate_fn)
            with torch.no_grad():
                evals.append(eval_f(model, ds, vocabulary=special_tokens, answer_position=-1, is_cot=is_cot))

    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to("cpu")

    return evals