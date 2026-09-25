import concurrent
import copy
import random
from concurrent.futures.process import ProcessPoolExecutor
from collections import defaultdict

from datasets import Dataset

from dataset.heuristics import counterfactual_heuristics, solve, is_reachable
from dataset.solvers import *

direct_answer = 200
cot_answer = 201

special_tokens = {1: 210, 0: 211}
pad = 208

tokenize_fact = lambda x: [(int(x), [null_e, fact_e])]
tokenize_rule = lambda x, r: [(int(y), [rule_e, rulestart_e]) for y in x] + [(int(r), [rule_e, ruleend_e])]


def tokenize(blob, order="facts_rules"):
    query = [(int(blob["query"][x]), [null_e, query_e]) for x in range(len(blob["query"]))]
    facts = [tokenize_fact(x) for x in blob["facts"]]
    rules = [tokenize_rule(x, r) for x, r in blob["rules"]]

    if order == "facts_rules":
        return query, facts + rules
    elif order == "rules_facts":
        return query, rules + facts
    elif order == "random":
        combined = facts + rules
        random.shuffle(combined)
        return query, combined


def type_tokenize(blob, expand=False, return_parts=["direct", "cot", "full_cot"],
                  solver_include_copy=False, solver_eager=True, solver_order="facts_rules"):
    result = {}

    processed_blob = process_labels(blob, expand=expand)
    query_tuples, problem_tuples = tokenize(blob=processed_blob, order=solver_order)
    input_tuples = [token_tuple for part in problem_tuples for token_tuple in part]
    input_ids = [x for x, y in input_tuples]
    input_types = [y for x, y in input_tuples]

    result["input"] = {
        "ids": input_ids,
        "type_embeddings": input_types,
        "labels": [-100] * len(input_ids)
    }

    if "direct" in return_parts:
        query_ids = [x for x, y in query_tuples]
        query_types = [y for x, y in query_tuples]
        answer_labels = [special_tokens[int(x)] for x in processed_blob["label"]]

        result["direct"] = {
            "ids": query_ids,
            "type_embeddings": query_types,
            "labels": answer_labels
        }

    if "cot" in return_parts:
        solved_sequence, _ = solve(blob["facts"], blob["rules"], blob["query"],
                                   break_early=True,
                                   include_copy=solver_include_copy,
                                   eager=solver_eager)

        cot_tuples = [(int(step), [null_e, fact_e]) for step in solved_sequence]
        full_sequence_tuples = query_tuples + cot_tuples
        cot_labels = [int(step) for step in solved_sequence]
        answer_labels = [special_tokens[int(x)] for x in processed_blob["label"]]
        result["cot"] = {
            "ids": [x for x, y in full_sequence_tuples],
            "type_embeddings": [y for x, y in full_sequence_tuples],
            "labels": cot_labels + answer_labels
        }

    if "full_cot" in return_parts:
        full_solved_sequence, _ = solve(blob["facts"], blob["rules"], blob["query"],
                                        break_early=False,
                                        include_copy=solver_include_copy,
                                        eager=solver_eager)
        full_cot_tuples = [(int(step), [null_e, fact_e]) for step in full_solved_sequence]
        full_sequence_tuples = query_tuples + full_cot_tuples
        full_cot_labels = [int(step) for step in full_solved_sequence]
        answer_labels = [special_tokens[int(x)] for x in processed_blob["label"]]
        result["full_cot"] = {
            "ids": [x for x, y in full_sequence_tuples],
            "type_embeddings": [y for x, y in full_sequence_tuples],
            "labels": full_cot_labels + answer_labels
        }

    return result


def process_labels(blob, expand=False):
    new_blob = copy.copy(blob)
    if expand:
        new_blob["query"] = blob["preds"]
        new_blob["label"] = [is_reachable(blob["facts"], blob["rules"], query) for query in blob["preds"]]
    else:
        new_blob["query"] = [blob["query"]]
        new_blob["label"] = [blob["label"]]
    return new_blob


def revert_rules_format(example):
    example['rules'] = [[premises, conclusion] for premises, conclusion in zip(example['rules']['premises'], example['rules']['conclusion'])]
    example['id'] = random.randint(0, 1_000_000_000)
    return example


def process(dataset, max_length=None, expand=False, solver_include_copy=False, solver_eager=True, solver_order="facts_rules"):
    ds = []
    for elem in dataset:
        result = type_tokenize(elem, expand=expand, solver_include_copy=solver_include_copy, solver_eager=solver_eager, solver_order=solver_order)
        result["depth"] = elem["depth"]
        result["extra_data"] = {}
        result["id"] = elem.get("id", random.randint(0, 1_000_000_000))

        if not max_length or len(result["input"]["ids"]) < max_length:
            ds.append(result)
    return ds


def get_reward_info(elem):
    reward_info = {
        "rl": True,
        "label": elem["full_cot"]["labels"][-1],
        "path": elem["full_cot"]["ids"][1:]
    }
    return reward_info


def train_curriculum(ds: Dataset, select_by_depth=None, eager=True, heuristics=[], heuristic_placement="replace" or "append",
                     balance=False, preserve_depth=False, preprocess_fn=revert_rules_format):
    if not eager:
        print("Shuffling dataset for random sampling...")
        ds = ds.shuffle()

    train_ds = []
    if select_by_depth is None:
        print("No curriculum specified, using all samples...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            train_ds = list(executor.map(preprocess_fn, ds.to_list()))
    else:
        print("Selecting samples based on curriculum...")
        items_to_process = []
        remaining_counts = select_by_depth.copy()
        total_required = sum(remaining_counts.values())

        for elem in ds:
            if len(items_to_process) == total_required:
                break
            depth = elem.get("depth")
            if remaining_counts.get(depth, 0) > 0:
                items_to_process.append(elem)
                remaining_counts[depth] -= 1

        print(f"Processing {len(items_to_process)} selected samples...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            train_ds = list(executor.map(preprocess_fn, items_to_process))

    if heuristics:
        heuristics = counterfactual_heuristics(train_ds, heuristics, balance=balance, preserve_depth=preserve_depth)
        if heuristic_placement == "replace":
            return heuristics
        elif heuristic_placement == "append":
            train_ds.extend(heuristics)
        else:
            raise RuntimeError(f"unknown: {heuristic_placement}")
    return train_ds

def reduce_ds(ds, fraction):
    if fraction >= 1.0:
        return ds

    unique_ids = set(elem["id"] for elem in ds)
    sorted_ids = sorted(list(unique_ids))
    target_idx = int(len(sorted_ids) * fraction)
    if target_idx == 0:
        return []

    threshold_id = sorted_ids[target_idx - 1]
    allowed_ids = set(i for i in unique_ids if i <= threshold_id)
    print(f"reduce_ds total: {len(unique_ids)} allowed: {len(allowed_ids)} cutoff: {threshold_id}")
    return [elem for elem in ds if elem["id"] in allowed_ids]

def finalize_ds(ds, custom_data_shuffle=False):
    if custom_data_shuffle:
        grouped_by_id = defaultdict(list)
        for elem in ds:
            grouped_by_id[elem['id']].append(elem)
        unique_ids = list(grouped_by_id.keys())
        random.shuffle(unique_ids)

        shuffled_ds = []
        for id_key in unique_ids:
            group = grouped_by_id[id_key]
            random.shuffle(group)
            shuffled_ds.extend(group)
        ds = shuffled_ds

    samples_by_depth = {}
    for elem in ds:
        if elem["depth"] not in samples_by_depth:
            samples_by_depth[elem["depth"]] = 0
        samples_by_depth[elem["depth"]] += 1
    print(f"len={len(ds)} distribution={samples_by_depth}")
    return ds


def prepare_ds_train(raw_ds, rl_frac=0, custom_data_shuffle=False, corrective_cot=False, cot=False, direct=False):
    ds = []
    grpo_count = 0
    corrective_cot_count = 0
    cot_count = 0
    direct_count = 0
    for data in raw_ds:

        if random.random() < rl_frac:
            grpo_count += 1
            ds.append({"depth": data["depth"],
                       "id": data["id"],
                       "extra_data": get_reward_info(data),
                       "input_ids": data["input"]["ids"] + [cot_answer] + data["direct"]["ids"],
                       "type_embeddings": data["input"]["type_embeddings"] + [[null_e, task_e]] + data["direct"]["type_embeddings"]})
            continue

        if corrective_cot:
            corrective_cot_count += 1
            ds.append({"depth": data["depth"],
                       "id": data["id"],
                       "input_ids": data["input"]["ids"] + [direct_answer] + data["direct"]["ids"] + [cot_answer] + data["cot"]["ids"],
                       "type_embeddings": data["input"]["type_embeddings"] + [[null_e, task_e]] + data["direct"]["type_embeddings"] + [[null_e, task_e]] + data["cot"]["type_embeddings"],
                       "labels": data["input"]["labels"] + [-100] + data["direct"]["labels"] + [-100] + data["cot"]["labels"]})

        if cot:
            cot_count += 1
            ds.append({"depth": data["depth"],
                       "id": data["id"],
                       "input_ids": data["input"]["ids"] + [cot_answer] + data["cot"]["ids"],
                       "type_embeddings": data["input"]["type_embeddings"] + [[null_e, task_e]] + data["cot"]["type_embeddings"],
                       "labels": data["input"]["labels"] + [-100] + data["cot"]["labels"]})

        if direct:
            direct_count += 1
            ds.append({"depth": data["depth"],
                       "id": data["id"],
                       "input_ids": data["input"]["ids"] + [direct_answer] + data["direct"]["ids"],
                       "type_embeddings": data["input"]["type_embeddings"] + [[null_e, task_e]] + data["direct"]["type_embeddings"],
                       "labels": data["input"]["labels"] + [-100] + data["direct"]["labels"]})

    ds = finalize_ds(ds, custom_data_shuffle=custom_data_shuffle)
    print(f"counts> grpo={grpo_count} corrective_cot={corrective_cot_count} cot={cot_count} direct={direct_count}")
    return ds


def prepare_ds_inference(raw_ds, is_cot=False):
    ds = []
    for data in raw_ds:
        ds.append({
            "depth": data["depth"],
            "id": data["id"],
            "input_ids": data["input"]["ids"] + [cot_answer if is_cot else direct_answer] + data["direct"]["ids"],
            "type_embeddings": data["input"]["type_embeddings"] + [[null_e, task_e]] + data["direct"]["type_embeddings"],
            "labels": data["input"]["labels"] + [-100] + data["cot" if is_cot else "direct"]["labels"],
            "full_labels": data["full_cot"]["labels"],
        })
    ds = finalize_ds(ds)
    return ds


if __name__ == '__main__':
    # r1 = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")["validation_lp"]
    # t1 = train_curriculum(r1, heuristics=["q", "r", "r2", "f", "rr"], balance=True, algo="LP")
    # p1 = process(t1, do_mask=False, cot=True)
    # for i in range(min(1, len(p1))):
    #     print(r1[i])
    #     print("input_ids: ", p1[i]["input_ids"])
    #     print("type_embeddings: ", p1[i]["type_embeddings"])
    #     print("labels: ", p1[i]["labels"])
    #     print()
    # print(len(p1))

    random.seed(123)
    # single query masking with heuristics
    ds1 = [
        {'preds': ['0', '1', '2', '3', '4'], 'rules': [[['1'], '2']], 'facts': ['0', '1'], 'query': '3', 'label': 0, 'depth': 2},
        {'preds': ['0', '1', '2', '3', '4'], 'rules': [[['1', '2'], '3']], 'facts': ['0', '1'], 'query': '3', 'label': 0, 'depth': 2},
        {'preds': ['0', '1', '2', '3', '4'], 'rules': [[['1'], '2'], [['1', '2'], '4']], 'facts': ['0', '1'], 'query': '4', 'label': 1, 'depth': 2},
        {'preds': ['0', '1', '2', '3', '4'], 'rules': [[['1'], '2'], [['1'], '3'], [['2', '3'], '4']], 'facts': ['1'], 'query': '2', 'label': 1, 'depth': 1},
    ]
    qp0, qp1, qp2, qp5 = process(copy.deepcopy(ds1))
    assert qp0["input"]["ids"] + qp0["direct"]["ids"] == [0, 1, 1, 2, 3]
    assert qp0["input"]["labels"] + qp0["direct"]["labels"] == [-100, -100, -100, -100, 211]
    assert qp0["input"]["type_embeddings"] + qp0["direct"]["type_embeddings"] == [[0, 1], [0, 1], [3, 4], [3, 5], [0, 2]]

    assert qp1["input"]["ids"] + qp1["direct"]["ids"] == [0, 1, 1, 2, 3, 3]
    assert qp1["input"]["labels"] + qp1["direct"]["labels"] == [-100, -100, -100, -100, -100, 211]
    assert qp1["input"]["type_embeddings"] + qp1["direct"]["type_embeddings"] == [[0, 1], [0, 1], [3, 4], [3, 4], [3, 5], [0, 2]]

    assert qp2["input"]["ids"] + qp2["direct"]["ids"] == [0, 1, 1, 2, 1, 2, 4, 4]
    assert qp2["input"]["labels"] + qp2["direct"]["labels"] == [-100, -100, -100, -100, -100, -100, -100, 210]
    assert qp2["input"]["type_embeddings"] + qp2["direct"]["type_embeddings"] == [[0, 1], [0, 1], [3, 4], [3, 5], [3, 4], [3, 4], [3, 5], [0, 2]]

    assert qp0["input"]["ids"] + qp0["cot"]["ids"] == [0, 1, 1, 2, 3, 2]
    assert qp0["input"]["labels"] + qp0["cot"]["labels"] == [-100, -100, -100, -100, 2, 211]
    assert qp0["input"]["type_embeddings"] + qp0["cot"]["type_embeddings"] == [[0, 1], [0, 1], [3, 4], [3, 5], [0, 2], [0, 1]]

    assert qp1["input"]["ids"] + qp1["cot"]["ids"] == [0, 1, 1, 2, 3, 3]
    assert qp1["input"]["labels"] + qp1["cot"]["labels"] == [-100, -100, -100, -100, -100, 211]
    assert qp1["input"]["type_embeddings"] + qp1["cot"]["type_embeddings"] == [[0, 1], [0, 1], [3, 4], [3, 4], [3, 5], [0, 2]]

    assert qp2["input"]["ids"] + qp2["cot"]["ids"] == [0, 1, 1, 2, 1, 2, 4, 4, 2, 4]
    assert qp2["input"]["labels"] + qp2["cot"]["labels"] == [-100, -100, -100, -100, -100, -100, -100, 2, 4, 210]
    assert qp2["input"]["type_embeddings"] + qp2["cot"]["type_embeddings"] == [[0, 1], [0, 1], [3, 4], [3, 5], [3, 4], [3, 4], [3, 5], [0, 2], [0, 1], [0, 1]]

    assert qp5["input"]["ids"] + qp5["cot"]["ids"] == [1, 1, 2, 1, 3, 2, 3, 4, 2, 2]
    assert qp5["input"]["labels"] + qp5["cot"]["labels"] == [-100, -100, -100, -100, -100, -100, -100, -100, 2, 210]
    assert qp5["input"]["type_embeddings"] + qp5["cot"]["type_embeddings"] == [[0, 1], [3, 4], [3, 5], [3, 4], [3, 5], [3, 4], [3, 4], [3, 5], [0, 2], [0, 1]]

    assert qp5["input"]["ids"] + qp5["full_cot"]["ids"] == [1, 1, 2, 1, 3, 2, 3, 4, 2, 2, 3, 4]
    assert qp5["input"]["labels"] + qp5["full_cot"]["labels"] == [-100, -100, -100, -100, -100, -100, -100, -100, 2, 3, 4, 210]
    assert qp5["input"]["type_embeddings"] + qp5["full_cot"]["type_embeddings"] == [[0, 1], [3, 4], [3, 5], [3, 4], [3, 5], [3, 4], [3, 4], [3, 5], [0, 2], [0, 1], [0, 1], [0, 1]]

    cp0, cp1, cp2, cp5 = process(copy.deepcopy(ds1), solver_include_copy=True)
    assert cp0["input"]["ids"] + cp0["cot"]["ids"] == [0, 1, 1, 2, 3, 0, 1, 2]
    assert cp0["input"]["labels"] + cp0["cot"]["labels"] == [-100, -100, -100, -100, 0, 1, 2, 211]
    assert cp0["input"]["type_embeddings"] + cp0["cot"]["type_embeddings"] == [[0, 1], [0, 1], [3, 4], [3, 5], [0, 2], [0, 1], [0, 1], [0, 1]]

    mixed_ds = prepare_ds_train([qp0], cot=True, direct=True, corrective_cot=True)
    assert id(mixed_ds[0]) != id(mixed_ds[1])
    assert id(mixed_ds[1]) != id(mixed_ds[2])
    assert id(mixed_ds[0]) != id(mixed_ds[2])

    has_cot, has_direct, has_corrective = False, False, False
    for item in mixed_ds:
        ids = item["input_ids"]
        if direct_answer in ids and cot_answer in ids:
            has_corrective = True
        elif cot_answer in ids:
            has_cot = True
        elif direct_answer in ids:
            has_direct = True
    assert has_cot and has_direct and has_corrective
