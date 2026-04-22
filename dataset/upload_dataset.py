import json

from conf import *
from datasets import Dataset, DatasetDict
from datasets import Features, Value, Sequence

schema = Features({
    'preds': Sequence(Value(dtype='string')),
    'rules': Sequence({
        'premises': Sequence(Value(dtype='string')),
        'conclusion': Value(dtype='string')
    }),
    'facts': Sequence(Value(dtype='string')),
    'query': Value(dtype='string'),
    'label': Value(dtype='int64'),
    'depth': Value(dtype='int64')
})


def process_and_generate_examples(filepath, cache_buster=None):
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)

                processed_rules = []
                if data.get('rules'):
                    for rule in data['rules']:
                        if isinstance(rule, list) and len(rule) == 2 and isinstance(rule[0], list):
                            processed_rules.append({"premises": rule[0], "conclusion": rule[1]})
                        else:
                            raise ValueError("Malformed rule structure")
                data['rules'] = processed_rules
                yield data

            except (ValueError, TypeError, IndexError) as e:
                print(f"Skipping malformed row: {e}\n{line.strip()}")
                continue


data_files = {
    "train_lp_downsampled": "/media/e/data/experiments/datasets/predicate_logic/train/700k/prop_examples_lp.jsonl",
    "train_rp_balanced_downsampled": "/media/e/data/experiments/datasets/predicate_logic/train/700k/prop_examples_rp_balanced_downsampled.jsonl",
    "validation_lp": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp.jsonl",
    "validation_lp_balanced": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_balanced.jsonl",
    "validation_lp_balanced_deep_30_pred": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_balanced_deep.jsonl",
    "validation_lp_balanced_deep_60_pred": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_balanced_deep_60.jsonl",
    "validation_lp_star_balanced_deep_30_pred": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_star_balanced_deep.jsonl",
    "validation_lp_star_balanced_deep_60_pred": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_lp_star_balanced_deep_60.jsonl",
    "validation_rp": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp.jsonl",
    "validation_rp_balanced": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced.jsonl",
    "validation_rp_balanced_deep_30_pred": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_deep.jsonl",
    "validation_rp_balanced_deep_60_pred": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_deep_60.jsonl",
    "validation_rp_balanced_3_premise": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_3_premise.jsonl",
    "validation_rp_balanced_2_premise": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_2_premise.jsonl",
    "validation_rp_balanced_1_premise": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_1_premise.jsonl",
    "validation_rp_balanced_1_2_premise": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_1+2_premise.jsonl",
    "validation_rp_balanced_2_3_premise": "/media/e/data/experiments/datasets/predicate_logic/validation/prop_examples_rp_balanced_2+3_premise.jsonl",
}

dataset_dict = DatasetDict({
    split_name: Dataset.from_generator(
        process_and_generate_examples,
        gen_kwargs={"filepath": file_path, "cache_buster": 1},
        features=schema
    )
    for split_name, file_path in data_files.items()
})

print("✅ DatasetDict created successfully:")
dataset_dict.push_to_hub("p12315132/computational_limits_of_implicit_deductive_reasoning")
print(f"Dataset successfully pushed!")
