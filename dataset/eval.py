import torch


def eval_model(llm_model, inference_ids, vocabulary, answer_position=-1, is_cot=False, kv_cache=True, log=lambda x: print(x, end=""), **kwargs):
    device = next(llm_model.parameters()).device
    correct = 0
    ood = 0
    false_positive = 0
    false_negative = 0
    positive = 0
    negative = 0
    correct_by_depth = {}
    incorrect_by_depth = {}

    for blob in inference_ids:
        depth = blob["depth"].item()
        expected = blob["labels"][0][answer_position].item()

        with torch.no_grad():
            generated_sequence = llm_model.generate(
                input_ids=blob["input_ids"].to(device),
                type_embeddings=blob["type_embeddings"].to(device),
                max_new_tokens=(64 if is_cot else 1),  # up dp 64 (deep validation dataset) preds
                kv_cache=kv_cache,
                stop_tokens=list(vocabulary.values()))
            next_token_id = generated_sequence[0, -1].item()

        if next_token_id in vocabulary.values():
            if next_token_id == vocabulary[1]:
                positive += 1
            else:
                negative += 1

        if next_token_id not in vocabulary.values():
            ood += 1
        elif expected == next_token_id:
            correct += 1
            correct_by_depth.setdefault(depth, 0)
            correct_by_depth[depth] += 1
        else:
            incorrect_by_depth.setdefault(depth, 0)
            incorrect_by_depth[depth] += 1
            if next_token_id == vocabulary[1]:
                false_positive += 1
            else:
                false_negative += 1

    log(f"positive predictions: {positive}, negative predictions: {negative}\n")
    log(f"correct: {correct}, false positive: {false_positive}, false negative: {false_negative}, ood: {ood}\n")
    log(f"correct by depth: {correct_by_depth}\n")
    log(f"incorrect by depth: {incorrect_by_depth}\n")
    return correct_by_depth
