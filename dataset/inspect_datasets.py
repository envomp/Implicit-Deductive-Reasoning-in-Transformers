import pandas as pd
from datasets import load_dataset
from processor import revert_rules_format

def calculate_average_premise_per_rule(datasets, target_depth=6):
    full_dataset_dict = load_dataset("p12315132/computational_limits_of_implicit_deductive_reasoning")
    for dataset_name in datasets:
        split_name = f"validation_{dataset_name}"
        data = [revert_rules_format(x) for x in full_dataset_dict[split_name]]

        all_arities = []
        for ex in data:
            if ex["depth"] == target_depth:
                rules = ex["rules"]
                if not rules:
                    continue
                example_avg_arity = sum(len(premises) for premises, _ in rules) / len(rules)
                all_arities.append(example_avg_arity)

        if all_arities:
            avg_arity = sum(all_arities) / len(all_arities)
            print(f"{dataset_name:<30} | {avg_arity:<15.2f}")
        else:
            print(f"{dataset_name:<30} | {'No data (d<=6)':<15}")



datasets = ["lp_balanced", "rp_balanced", "rp_balanced_3_premise", "rp_balanced_2_premise", "rp_balanced_1_premise", "rp_balanced_1_2_premise", "rp_balanced_2_3_premise"]
calculate_average_premise_per_rule(datasets)
