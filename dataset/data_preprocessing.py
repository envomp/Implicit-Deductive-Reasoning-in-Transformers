from torch.utils.data import default_collate
import torch
from typing import Literal


def next_word_prediction_labels(input_ids, answer_ids, strategy: Literal["answer_only", "all", "masked"] = "answer_only"):
    """
    > strategy = "answer_only", input_ids = i n p u t Q, answer_ids = V
        i n p u t Q
        - - - - - V

    > strategy = "all", input_ids = i n p u t Q, answer_ids = V
        i n p u t Q
        n p u t Q V

    > strategy = "masked", input_ids = i n p u t Q1 Q2 Q3, answer_ids = V1 V2 V3
        i n p u t Q1 Q2 Q3
        - - - - - V1 V2 V3
    """
    if strategy == "answer_only":
        input = input_ids + answer_ids
        labels = [-100 for _ in input_ids] + answer_ids
        input.pop(-1)
        labels.pop(0)
        return labels, input

    if strategy == "all":
        input = input_ids + answer_ids
        labels = input_ids + answer_ids
        input.pop(-1)
        labels.pop(0)
        return labels, input

    if strategy == "masked":
        input = input_ids
        labels = [-100 for _ in range(len(input_ids) - len(answer_ids))] + answer_ids
        return labels, input

    raise ValueError(f"unknown strategy: {strategy}")


def pad_collate(batch, padding):
    max_size = max([len(x["input_ids"]) for x in batch])
    collated = []
    extra_data = []

    for elem in batch:
        copy = {}
        extra = elem.get("extra_data", {})
        if extra.get("rl", False):
            copy["input_ids"] = torch.tensor([padding] * (max_size - len(elem["input_ids"])) + elem["input_ids"])
            copy["labels"] = torch.full((max_size,), -100, dtype=torch.long)
            if elem.get("type_embeddings") is not None:
                copy["type_embeddings"] = torch.tensor([[0, 0]] * (max_size - len(elem["type_embeddings"])) + elem["type_embeddings"])
        else:
            copy["input_ids"] = torch.tensor(elem["input_ids"] + [padding] * (max_size - len(elem["input_ids"])))
            copy["labels"] = torch.tensor(elem["labels"] + [-100] * (max_size - len(elem["labels"])))
            if elem.get("type_embeddings") is not None:
                copy["type_embeddings"] = torch.tensor(elem["type_embeddings"] + [[0, 0]] * (max_size - len(elem["type_embeddings"])))

        if elem.get("depth") is not None:
            copy["depth"] = elem["depth"]

        extra_data.append(extra)
        collated.append(copy)

    ds = default_collate(collated)
    ds["extra_data"] = extra_data
    return ds


def labels_to_one_hot(labels, num_classes):
    batch_size, seq_len = labels.shape
    one_hot = torch.zeros(batch_size, seq_len, num_classes, dtype=labels.dtype, device=labels.device)
    valid_indices = labels != -100
    one_hot[valid_indices, labels[valid_indices]] = 1
    return one_hot


def create_uniform_soft_labels(target_groups, num_classes):
    """
    Transforms groups of target indices into uniform probability distributions.
    For each group, the probability is 1/n, distributed among the n members.
    """
    batch_size = len(target_groups)
    soft_labels = torch.zeros(batch_size, num_classes, device='cpu')

    for i, group in enumerate(target_groups):
        if isinstance(group, int) and group == -100 or not group:
            continue

        if isinstance(group, int):
            group = [group]

        n = len(group)
        if n > 0:
            prob = 1.0 / n
            for label_index in group:
                if 0 <= label_index < num_classes:
                    soft_labels[i, label_index] = prob
    return soft_labels


if __name__ == '__main__':
    labels, input_seq = next_word_prediction_labels([1, 2, 3], [4, 5], "answer_only")
    assert labels == [-100, -100, 4, 5]
    assert input_seq == [1, 2, 3, 4]

    labels, input_seq = next_word_prediction_labels([1, 2, 3], [4, 5], "all")
    assert labels == [2, 3, 4, 5]
    assert input_seq == [1, 2, 3, 4]

    labels, input_seq = next_word_prediction_labels([1, 2, 3, 4, 5, 6], [7, 8, 9], "masked")
    assert labels == [-100, -100, -100, 7, 8, 9]
    assert input_seq == [1, 2, 3, 4, 5, 6]

    target_indices_groups = [-100, [0], 1, [2, 3], [210]]
    soft_labels = create_uniform_soft_labels(target_indices_groups, 5)
    print(soft_labels)
